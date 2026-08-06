"""Continued Pretraining and supervised fine-tuning training.

"""
__author__ = 'Paul Landes'
from typing import Any, ClassVar
from dataclasses import dataclass, field
from abc import ABCMeta, abstractmethod
import logging
import sys
import copy as cp
from pathlib import Path
from io import TextIOBase, StringIO
import json
from datasets import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizer, TrainingArguments
from transformers.trainer_utils import TrainOutput
from peft import PeftModelForCausalLM
from zensols.util.time import time
from zensols.config import Dictable, Configurable
from zensols.persist import PersistedWork, Primeable, persisted
from .task import TaskDatasetFactory

logger = logging.getLogger(__name__)


@dataclass
class TrainerResource(Dictable, Primeable, metaclass=ABCMeta):
    """Configures and instantiates the base mode, PEFT mode, and the tokenizer.

    """
    model_args: dict[str, Any] = field(default=None)
    """The parameters that create the base model and tokenzier."""

    cache: bool = field(default=True)
    """Whether to cache the tokenizer and model."""

    def __post_init__(self):
        self._model_tokenizer_pw = PersistedWork(
            '_model_tokenizer_pw', self, cache_global=self.cache)
        self._peft_model_pw = PersistedWork(
            '_peft_model_pw', self, cache_global=self.cache)

    @abstractmethod
    def _create_model_tokenizer(self) -> \
            tuple[PreTrainedTokenizer, PreTrainedModel]:
        pass

    @abstractmethod
    def _create_peft_model(self) -> PeftModelForCausalLM:
        pass

    @property
    @persisted('_model_tokenizer_pw')
    def _model_tokenizer(self) -> tuple[Any, Any]:
        return self._create_model_tokenizer()

    @property
    def model(self) -> PreTrainedModel:
        """The base model."""
        return self._model_tokenizer[0]

    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        """The base tokenizer."""
        return self._model_tokenizer[1]

    @property
    @persisted('_peft_model_pw')
    def peft_model(self) -> PeftModelForCausalLM:
        """The PEFT (Parameter-Efficient Fine-Tuning) such as LoRA."""
        return self._create_peft_model()

    def prime(self):
        super().prime()
        self.peft_model


@dataclass(repr=False)
class ModelResult(Dictable):
    """The trained model config, location and configuration used to train it.

    """
    _DICTABLE_ATTRIBUTES: ClassVar[set[str]] = frozenset(
        'global_step training_loss metrics'.split())

    train_output: TrainOutput = field(repr=False)
    """The output returned from the trainer."""

    output_dir: Path = field(default=None)
    """The directory of the models checkpoints."""

    train_params: dict[str, Any] = field(default=None)
    """The training parameters used to configure the trainer."""

    config: Configurable = field(default=None)
    """The application configuration used to configure the trainer."""

    @property
    def global_step(self) -> int:
        """The global step from :obj:`train_output`."""
        return self.train_output.global_step

    @property
    def training_loss(self) -> float:
        """The training loss from :obj:`train_output`."""
        return self.train_output.training_loss

    @property
    def metrics(self) -> dict[str, float]:
        """Training metrics from :obj:`train_output`."""
        return self.train_output.metrics

    def _from_dictable(self, *args, **kwargs) -> dict[str, Any]:
        targs: TrainingArguments = self.train_params['args']
        dct: dict[str, Any] = super()._from_dictable(*args, **kwargs)
        dct = cp.deepcopy(dct)
        dct['train_params'].pop('args')
        dct['train_params']['args'] = json.loads(targs.to_json_string())
        dct['config'] = self.config.asdict()
        return dct

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout,
              include_training_arguments: bool = False,
              include_config: bool = False):
        dct: dict[str, Any] = cp.deepcopy(self.asdict())
        # move long params to end since now dicts are stable ordered
        for key in 'train_params config'.split():
            dct[key] = dct.pop(key)
        if not include_training_arguments:
            dct['train_params'].pop('args')
        if not include_config:
            dct.pop('config')
        self._write_object(dct, depth, writer)

    def __str__(self) -> str:
        return f'global step: {self.global_step}, loss: {self.training_loss}'


@dataclass
class Trainer(Dictable, metaclass=ABCMeta):
    """A configurable supervised fine-tuning trainer wrapper.

    """
    config: Configurable = field()
    """Used to save to the model result."""

    resource: TrainerResource = field()
    """Used to create the model and tokenizer."""

    train_params: dict[str, Any] = field()
    """The training parameters used to configure the trainer."""

    eval_params: dict[str, Any] = field()
    """The evaluation parameters used to configure the trainer."""

    train_source: TaskDatasetFactory = field()
    """A factory that creates new datasets used to train using this instance."""

    eval_source: TaskDatasetFactory = field()
    """A factory that creates new datasets used to evaluation."""

    peft_output_dir: str | Path = field()
    """The directory in which to save the PEFT adapter."""

    merged_output_dir: str | Path = field()
    """The directory in which to save the base model with the PEFT adapter
    merged.

    """
    def __post_init__(self):
        for attr in 'peft_output_dir merged_output_dir'.split():
            val = getattr(self, attr)
            if isinstance(val, str):
                setattr(self, attr, Path(val))

    def _get_training_params(self) -> dict[str, Any]:
        from trl import SFTConfig
        params: dict[str, Any] = cp.deepcopy(self.train_params)
        args: SFTConfig = params['args']
        assert isinstance(args, SFTConfig)
        assert hasattr(args, 'dataset_text_field')
        args.dataset_text_field = self.train_source.text_field
        if self.eval_source is not None:
            params['eval_dataset'] = self.eval_source.create()
            args.__dict__.update(self.eval_params)
        return params

    @abstractmethod
    def _train(self, params: dict[str, Any], train_ds: Dataset,
               eval_ds: Dataset = None) -> TrainOutput:
        pass

    def train(self) -> ModelResult:
        """Train the model."""
        params: dict[str, Any] = self._get_training_params()
        train_dataset: Dataset = self.train_source.create()
        checkpoint_dir = Path(params['args'].output_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.peft_output_dir.mkdir(parents=True, exist_ok=True)
        if self.merged_output_dir is not None:
            self.merged_output_dir.mkdir(parents=True, exist_ok=True)
        with time('model trained'):
            output: TrainOutput = self._train(params, train_dataset)
            result: ModelResult = ModelResult(output)
            result.output_dir = self.peft_output_dir
            result.train_params = params
            result.config = self.config
            logger.info(f'training complete: {result}')
        return result

    def _from_dictable(self, *args, **kwargs) -> dict[str, Any]:
        dct: dict[str, Any] = super()._from_dictable(*args, **kwargs)
        dct['train_params'] = self._get_training_params()
        return dct

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout,
              include_training_arguments: bool = False):
        dct: dict[str, Any] = cp.deepcopy(self.asdict())
        args: TrainingArguments = dct['train_params'].pop('args')
        dct.pop('train_source')
        if include_training_arguments:
            sio = StringIO()
            self._write_block(str(args), depth=2, writer=sio)
            dct['train_params']['args'] = sio.getvalue().strip()
            dct.pop('config')
            self._write_object(dct, depth, writer)
        if self.train_source is not None:
            self._write_line('train_source:', depth, writer)
            self._write_object(self.train_source, depth + 1, writer)
