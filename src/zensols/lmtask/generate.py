"""Facade to HuggingFace text generation.

"""
from __future__ import annotations
__author__ = 'Paul Landes'
from typing import List, Tuple, Dict, Iterable, Any, Union, Type, ClassVar
from dataclasses import dataclass, field
from abc import ABCMeta, abstractmethod
import sys
import logging
import re
import collections
from threading import Thread
from pathlib import Path
from io import TextIOBase
import torch
from torch import Tensor
import textwrap as tw
from datasets import Dataset
from datasets.formatting.formatting import LazyBatch
from transformers import (
    AutoModel, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel,
    AutoModelForCausalLM, BatchEncoding, TextIteratorStreamer
)
from peft import AutoPeftModelForCausalLM
from zensols.util import time, Hasher
from zensols.persist import persisted, Stash, FileTextUtil
from zensols.config import Dictable, ConfigFactory
from . import TaskError, Task, TaskRequest, TaskResponse, TaskDatasetFactory

logger = logging.getLogger(__name__)


@dataclass
class _Resource(object):
    tokenizer: PreTrainedTokenizer = field(default=None)
    model: PreTrainedModel = field(default=None)


@dataclass
class GeneratorResource(Dictable):
    """A client facade of a chat-based large language model.

    """
    _MODEL_DESC_PAT: ClassVar[re.Pattern] = re.compile(r'^(?:.*\/)(.+)$')

    name: str = field()
    """The section of this configured instance in the application config."""

    model_id: Union[str, Path] = field()
    """The HF model ID or path to the model."""

    model_class: Type[AutoModel] = field(default=AutoModelForCausalLM)
    """The class used to create the model with
    :meth:`~transformers.AutoModel.from_pretrained`.

    """
    tokenizer_class: Type[AutoTokenizer] = field(default=AutoTokenizer)
    """The class used to create the tokenizer with
    :meth:`~transformers.AutoTokenizer.from_pretrained.

    """
    peft_model_id: Union[str, Path] = field(default=None)
    """The HF model ID or path to the Peft model or ``None`` if there is none.

    """
    peft_model_class: Type[AutoModel] = field(default=AutoPeftModelForCausalLM)
    """The class used to create the model with
    :meth:`~transformers.AutoModel.from_pretrained`.

    """
    model_desc: str = field(default=None)
    """A human readable description of the model this resource contains."""

    system_role_name: str = field(default='system')
    """The default name of the system's role."""

    model_args: Dict[str, Any] = field(default_factory=dict)
    """The arguments given to the HF model ``from_pretrained`` method.

    """
    def __post_init__(self):
        if isinstance(self.model_id, Path):
            self.model_id = str(self.model_id)
        if self.model_desc is None:
            self.model_desc = self._shorten_model_id(self.model_id)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'created generator: {self.name}')

    @classmethod
    def _shorten_model_id(cls: Type, model_id: str):
        return re.sub(cls._MODEL_DESC_PAT, r'\1', model_id)

    @classmethod
    def get_model_path(cls: Type, model_id: str, parent: Path = None) -> Path:
        """Create a normalized file name from a HF model ID string useful for
        creating checkpoint directory names.

        :param model_id: the model ID (i.e. ``meta-llama/Llama-3.1-8B``)

        :param parent: the base directory used in the return value if given

        """
        model_id = cls._shorten_model_id(model_id)
        path = Path(FileTextUtil.normalize_text(model_id))
        if parent is not None:
            path = parent / path
        return path

    @property
    def model_file_name(self) -> str:
        """A normalized file name friendly string based on :obj:`model_desc`."""
        return FileTextUtil.normalize_text(self.model_desc)

    @property
    @persisted('_resource_cache_pw', cache_global=True)
    def _resource_cache(self) -> Dict[str, _Resource]:
        return collections.defaultdict(_Resource)

    def configure_tokenizer(self, tokenizer: PreTrainedTokenizer):
        """Make any necessary updates programatically (i.e. set special
        tokens).

        """
        pass

    def configure_model(self, model: PreTrainedModel):
        """Make any necessary updates programatically."""
        pass

    def _load_tokenizer(self) -> PreTrainedTokenizer:
        model_id: str = self.model_id
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'creating tokenizer: {model_id}')
        params: Dict[str, Any] = {}
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'tokenizer params: {params}')
        tokenizer = self.tokenizer_class.from_pretrained(model_id, **params)
        self.configure_tokenizer(tokenizer)
        return tokenizer

    def _load_model(self) -> PreTrainedModel:
        model_id: str = self.model_id
        params: Dict[str, Any] = dict(self.model_args)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'model params: {params}')
        with time(f'loaded model: {model_id}', logging.DEBUG):
            cls: Type[AutoModel]
            if self.peft_model_id is None:
                cls = self.model_class
            else:
                cls = self.peft_model_class
                model_id = self.peft_model_id
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'loading {model_id}, generator: {self.name}')
                self.write_to_log(logger, logging.DEBUG)
            model = cls.from_pretrained(model_id, **params)
            self.configure_model(model)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'model type: {type(model)}')
        return model

    def clear(self, include_cuda: bool = True):
        """Clear the cached tokenizer, model and optionally CUDA."""
        self._resource_cache.clear()
        if include_cuda:
            torch.cuda.empty_cache()

    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        """The model's tokenzier."""
        res: _Resource = self._resource_cache[self.model_id]
        if res.tokenizer is None:
            res.tokenizer = self._load_tokenizer()
        return res.tokenizer

    @property
    @persisted('_model')
    def model(self) -> PreTrainedModel:
        """The LLM."""
        res: _Resource = self._resource_cache[self.model_id]
        if res.model is None:
            res.model = self._load_model()
        return res.model

    def __repr__(self):
        return self.model_id


@dataclass
class GeneratorOutput(Dictable):
    """Container instances of model output from :class:`.TextGenerator`.

    """
    model_output: str = field()
    """The unmodified raw model output."""

    parsed: Tuple[str, ...] = field()
    """The processed model output with special tokens stripped."""


@dataclass
class TextGenerator(Dictable, metaclass=ABCMeta):
    """A client facade of a chat-based large language model.

    """
    def __post_init__(self):
        pass

    @abstractmethod
    def _generate(self, prompt: str) -> GeneratorOutput:
        pass

    def generate(self, prompt: str) -> GeneratorOutput:
        """Generate a textual response (usually from a large langauge model)."""
        with time('generated response'):
            return self._generate(prompt)

    def clear(self):
        """Clear any model state."""
        pass


@dataclass
class ConstantTextGenerator(TextGenerator):
    """A generator that responses with :obj:`response` with every generation
    call for the purpose of debugging.

    """
    config_factory: ConfigFactory = field()
    """Used to set optional mock attributes in :obj:`post_init_source`."""

    response: str = field()
    """The fixed response for each :meth:`generate` call or the prompt if
    ``None``.

    """
    post_init_source: str = field(default=None)
    """Python source code to run in the initializer."""

    def __post_init__(self):
        super().__post_init__()
        if self.post_init_source is not None:
            exec(self.post_init_source)

    def _generate(self, prompt: str) -> GeneratorOutput:
        response: str = self.response
        if response is None:
            response = prompt
        return GeneratorOutput(response, response)


@dataclass
class ModelTextGenerator(TextGenerator):
    """An implementation that uses HuggingFace framework classes from
    :class:`.GeneratorResource` to answer queries.

    """
    resource: GeneratorResource = field()
    """The class that creates resources such as the tokenizer and model."""

    tokenize_params: Dict[str, Any] = field(
        default_factory=lambda: {'return_tensors': 'pt'})
    """Parameters to add or override in the model tokenize call."""

    tokenize_decode_params: Dict[str, Any] = field(default_factory=dict)
    """Parameters to add or override in the model tokenize call."""

    generate_params: Dict[str, Any] = field(default_factory=dict)
    """Parameters given to the model's inference method for each prompt."""

    def _process_output(self, input_ids: Tensor, model_output: Tensor) -> \
            Tensor:
        return model_output[0]

    def _parse_response(self, text: str) -> str:
        """Parse the model's output.

        :param text: the model prompt

        :return: the responses from the model

        """
        return text

    def _replace_output(self, text: str) -> str:
        """Make any necessary changes in ``text`` of the model output."""
        return text

    def _get_tokenize_params(self) -> Dict[str, Any]:
        return self.tokenize_params

    def _get_tokenize_decode_params(self) -> Dict[str, Any]:
        return self.tokenize_decode_params

    def _get_generate_params(self) -> Dict[str, Any]:
        mr: GeneratorResource = self.resource
        tokenizer: PreTrainedTokenizer = mr.tokenizer
        params: Dict[str, Any] = dict(self.generate_params)
        params.update(dict(
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id))
        return params

    def _generate(self, prompt: str) -> GeneratorOutput:
        """Generate a textual response using the language model."""
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'generating text for: <<{tw.shorten(prompt, 60)}>>')
        mr: GeneratorResource = self.resource
        tokenizer: PreTrainedTokenizer = mr.tokenizer
        model: PreTrainedModel = mr.model

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'prompt: <<{prompt}>>')
            logger.debug(f'tokenize_params: {self._get_tokenize_params()}')
            logger.debug('tokenize_decode_params: ' +
                         str(self._get_tokenize_decode_params()))
            logger.debug(f'generate_params: {self._get_generate_params()}')

        be: BatchEncoding = tokenizer(prompt, **self._get_tokenize_params())
        input_ids: Tensor = be.input_ids.to(model.device)
        if logger.isEnabledFor(logging.TRACE):
            input_text: str = tokenizer.decode(
                input_ids[0], **self._get_tokenize_decode_params())
            logger.trace(f'input text: <<{input_text}>>')
        with torch.no_grad():
            model_output: Tensor = model.generate(
                input_ids,
                **self._get_generate_params())
        model_output = self._process_output(input_ids, model_output)
        model_output_raw: str = tokenizer.decode(
            model_output, **self._get_tokenize_decode_params())
        if logger.isEnabledFor(logging.TRACE):
            logger.trace(f'model raw output: <<{model_output_raw}>>')
        response: str = self._parse_response(model_output_raw)
        return GeneratorOutput(model_output_raw, response)

    def stream(self, prompt: str, writer: TextIOBase = sys.stdout,
               width: int = 80):
        """Stream the model's output from a ``prompt`` input.

        :param prompt: the input to give to the model

        :param writer: the data sink

        :param width: the maximum width of each line's streamed text; if
                      ``None``, no modification will be done on the text output

        """
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'stream ouput of prompt: <<{prompt}>>')
        mr: GeneratorResource = self.resource
        tokenizer: PreTrainedTokenizer = mr.tokenizer
        model: PreTrainedModel = mr.model
        inputs: Tensor = tokenizer(prompt, **self._get_tokenize_params())
        inputs = inputs.to(model.device)
        text_streamer = TextIteratorStreamer(tokenizer)
        thread = Thread(
            target=model.generate,
            kwargs=dict(
                inputs=inputs.input_ids,
                streamer=text_streamer,
                **self._get_generate_params()))
        thread.start()
        cur_width: int = 0
        text: str
        for text in text_streamer:
            if width is not None:
                if text.endswith('\n\n'):
                    cur_width = 0
                else:
                    text = self._replace_output(text)
                    text = text.replace('\n', ' ')
                    cur_width += len(text) + 1
                    if cur_width >= width:
                        writer.write('\n')
                        cur_width = 0
            writer.write(text)
            writer.flush()
        if width is not None:
            writer.write('\n')
        writer.flush()
        thread.join()

    def clear(self):
        super().clear()
        self.resource.clear()


@dataclass
class ReplaceTextGenerator(ModelTextGenerator):
    """A text generator that generates response by replacing regular
    expressions.  This is helpful for removing special tokens.

    """
    replacements: Tuple[Tuple[Union[str, re.Pattern], str], ...] = \
        field(default=())
    """The a tuple ``(<regular expression>, <replacement>)`` to replace in the
    parsed output from the model.  String patters are compiled with
    :func:`re.compile`.

    """
    def __post_init__(self):
        def map_expr(expr: Tuple[Union[str, re.Pattern], str]):
            pat = expr[0]
            if isinstance(pat, str):
                try:
                    pat = re.compile(pat)
                except Exception as e:
                    raise TaskError(
                        "Could not compile task name regex: " +
                        f"'{pat}' in <<{self.replacements}>>") from e
            return (pat, expr[1])

        super().__post_init__()
        self.replacements = tuple(map(map_expr, self.replacements))

    def _replace_output(self, text: str) -> str:
        for expr, repl in self.replacements:
            text = re.sub(expr, repl, text)
        return text

    def _parse_response(self, text: str) -> Iterable[str]:
        return self._replace_output(text)


@dataclass
class CachingGenerator(TextGenerator):
    """A generator that caches response using a hash of the model input as a
    key.

    """
    _delegate: TextGenerator = field()
    """The generator used to create the response to cache and return."""

    _stash: Stash = field()
    """The stash that reuses the LLM responses."""

    _hasher: Hasher = field(default_factory=Hasher)
    """Usd to hash prompts to stash keys."""

    def _generate(self, prompt: str) -> GeneratorOutput:
        self._hasher.reset()
        self._hasher.update(prompt)
        key: str = self._hasher()
        inst: GeneratorOutput = self._stash.load(key)
        if inst is None:
            inst = self._delegate.generate(prompt)
            self._stash.dump(key, inst)
        return inst

    def clear(self):
        super().clear()
        self._delegate.clear()
        self._stash.clear()


@dataclass(repr=False)
class GenerateTask(Task):
    """Uses a :class:`.TextGenerator` (:obj:`generator`) to generate a response.

    """
    generator: TextGenerator = field()
    """A client facade of a chat or instruct-based large language model."""

    resource: GeneratorResource = field()
    """The class that creates resources such as the tokenizer and model.  This
    should be the base model resource so training tasks do not depend on the
    model they will eventually create.

    This is also used by :class:`.InstructTask` for its chat template.

    """
    train_add_eos: bool = field(default=False)
    """Whether to add the end of sentence token to the output when mapping the
    dataset for training.  Newer versions of the :class:`.trl.SFTTrainer` class
    add (and force) this already.

    """
    def _process(self, request: TaskRequest) -> TaskResponse:
        """Process a query (see :meth:`process`)."""
        request = self._prepare_request(request)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'processing generate task request: {request}')
        try:
            gen_output: GeneratorOutput = self.generator.generate(
                prompt=request.model_input)
            return self.response_class(
                request=request,
                model_output_raw=gen_output.model_output,
                model_output=gen_output.parsed)
        except Exception as e:
            msg: str = f'Could not process task {self.__class__}'
            raise TaskError(msg, request.model_input) from e

    def _prepare_dataset(self, ds: Dataset, factory: TaskDatasetFactory) -> \
            Dataset:
        """Instances of this class assume a text generation use case, and thus,
        add an end of sequence token to each row of the dataset.

        """
        def map_batch(batch: LazyBatch) -> Dict[str, List[Dict[str, Any]]]:
            return {field: list(map(lambda s: s + eos_token, batch[field]))}

        if self.train_add_eos:
            field: str = factory.text_field
            tokenizer: PreTrainedTokenizer = self.resource.tokenizer
            eos_token: str = tokenizer.eos_token
            ds = ds.map(map_batch, batched=True)
        return ds

    def clear(self):
        super().clear()
        self.generator.clear()
