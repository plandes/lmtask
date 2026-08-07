"""Continued Pretraining and supervised fine-tuning training.

"""
__author__ = 'Paul Landes'
from typing import Any, ClassVar
from dataclasses import dataclass, field
from abc import ABCMeta, abstractmethod
import logging
import sys
import pickle
import copy as cp
from time import perf_counter
from pathlib import Path
from io import TextIOBase, StringIO
import json
from datasets import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizer, TrainingArguments
from transformers.trainer_utils import TrainOutput
from peft import PeftModelForCausalLM
from zensols.util.fail import APIError
from zensols.util.time import time
from zensols.config import Dictable, Configurable
from zensols.persist import PersistedWork, Primeable, persisted
from .task import TaskDatasetFactory

logger = logging.getLogger(__name__)


class TrainError(APIError):
    pass


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
class TrainResult(Dictable):
    """The trained model config, location and configuration used to train it.

    """
    _DICTABLE_ATTRIBUTES: ClassVar[set[str]] = frozenset(
        'global_step training_loss metrics'.split())

    train_output: TrainOutput = field(repr=False)
    """The output returned from the trainer."""

    peft_output_dir: Path = field()
    """The directory of the models checkpoints."""

    train_params: dict[str, Any] = field()
    """The training parameters used to configure the trainer."""

    config: Configurable = field()
    """The application configuration used to configure the trainer."""

    time_elapsed: int = field()
    """Time in seconds it took to train."""

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

    peft_output_dir: Path = field()
    """The directory in which to save the PEFT adapter."""

    result_file: Path = field()
    """The file to save the training statistics for benchmarking."""

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

    @property
    def model_exists(self) -> bool:
        """Whether the trained model already exists."""
        return self.peft_output_dir.is_dir()

    def train(self) -> TrainResult:
        """Train the model."""
        if self.model_exists:
            raise TrainError('Can not overwrite existing model; ' +
                             f'delete first: {self.peft_output_dir}')
        params: dict[str, Any] = self._get_training_params()
        train_dataset: Dataset = self.train_source.create()
        checkpoint_dir = Path(params['args'].output_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.peft_output_dir.mkdir(parents=True, exist_ok=True)
        start: float = perf_counter()
        output: TrainOutput = self._train(params, train_dataset)
        time_elapsed: float = perf_counter() - start
        result: TrainResult = TrainResult(
            train_output=output,
            peft_output_dir=self.peft_output_dir,
            train_params=params,
            config=self.config,
            time_elapsed=time_elapsed)
        if logger.isEnabledFor(logging.INFO):
            logger.info(time.format_elapse('training finished', time_elapsed))
        return result

    def save_result(self, result: TrainResult):
        """Save the result to the file system."""
        result_path: Path = self.result_file
        result_path.parent.mkdir(parents=True, exist_ok=True)
        with open(result_path, 'wb') as f:
            pickle.dump(result, f)
        logger.info(f'wrote: {result_path}')

    def load_result(self) -> TrainResult:
        """Load the model results."""
        path: Path = self.result_file
        if not path.is_file():
            raise TrainError(
                f'It apperas the model exists but missing result file: {path}')
        with open(path, 'rb') as f:
            return pickle.load(f)

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
