"""Task implementations.

"""
__author__ = 'Paul Landes'

from typing import Any, Dict, List, Tuple, Union
from dataclasses import dataclass, field
import sys
import logging
import warnings
from pathlib import Path
from io import TextIOBase
from jinja2 import Template, Environment, BaseLoader
from torch import Tensor
from zensols.util import openread
from datasets.formatting.formatting import LazyBatch
from transformers import PreTrainedTokenizer
from datasets import Dataset
from .task import TaskError, TaskRequest, TaskDatasetFactory
from .generate import (
    GeneratorResource, ReplaceTextGenerator, GenerateTask
)

logger = logging.getLogger(__name__)


@dataclass
class InstructTaskRequest(TaskRequest):
    """A request that has a query portion to be added to the compiled prompt.

    """
    instruction: Any = field(default=None)
    """The instruction given to the model to complete the task.

    """
    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout,
              include_instruction: bool = True):
        super().write(depth, writer)
        if include_instruction:
            self._write_line('instruction:', depth, writer)
            if isinstance(self.instruction, str):
                self._write_block(self.instruction, depth + 1, writer)
            else:
                self._write_object(self.instruction, depth + 1, writer)

    def __str__(self) -> str:
        return f"{super().__str__()}, {self._shorten_value('instruction')}"


@dataclass
class NShotTaskRequest(InstructTaskRequest):
    """A request that adds training examples to the prompt.

    """
    examples: Tuple[Any, ...] = field(default=None)
    """The examples given for N-shot learning."""


@dataclass
class InstructModelTextGenerator(ReplaceTextGenerator):
    """A generator that uses instruct based models for inference.

    """
    def _process_output(self, input_ids: Tensor, model_output: Tensor) -> \
            Tensor:
        model_output = super()._process_output(input_ids, model_output)
        start_index: int = input_ids.shape[-1]
        input_removed_ids: Tensor = model_output[start_index:]
        if logger.isEnabledFor(logging.DEBUG):
            mr: GeneratorResource = self.resource
            tokenizer: PreTrainedTokenizer = mr.tokenizer
            original: str = tokenizer.decode(
                model_output, **self.tokenize_decode_params)
            removed: str = tokenizer.decode(
                model_output[:start_index], **self.tokenize_decode_params)
            input_removed: str = tokenizer.decode(
                input_removed_ids, **self.tokenize_decode_params)
            logger.debug(f'original: <<{original}>>')
            logger.debug(f'removing input: <<{removed}>>')
            logger.debug(f'input removed: <<{input_removed}>>')
        return input_removed_ids


@dataclass(repr=False)
class InstructTask(GenerateTask):
    """A task that is resolved using instructions given to the language model.

    **Important**: If :obj:`.InstructTaskRequest.model_input` is non-``None``
    that value is used verbatim and :obj:`.InstructTaskRequest.instruction` is
    ignored.

    """
    role: str = field(default='You are a helpful assistant.')
    """The role of the chat dialogue."""

    train_template: Union[str, Path] = field(
        default='### Question: {{ instruction }}\n### Answer: {{ output }}')
    """Used to create format the datasets training text :obj:`generator`."""

    inference_template: Union[str, Path] = field(
        default='{{request.instruction}}')
    """The instructions given to :obj:`generator`."""

    chat_template_args: Dict[str, Any] = field(default_factory=dict)
    """Arguments given to ``apply_chat_template``."""

    apply_chat_template: bool = field(default=True)
    """Whether format the prompt into one that conforms to the model's instruct
    syntax.

    """
    train_apply_chat_template: bool = field(default=False)
    """Like :obj:`apply_chat_template`, but whether to apply during training.
    If this is ``False``, a conversational ``messages`` with dictionary list is
    used instead.

    """
    def __post_init__(self):
        if self.train_add_eos:
            warnings.warn(
                message=("Field 'train_add_eos' is 'True' in InstructTask" +
                         "but not needed since it applies a chat template."),
                category=UserWarning)

    def _apply_messages(self, prompt: str) -> List[Dict[str, str]]:
        role_namme: str = self.resource.system_role_name
        return [
            {'role': role_namme, 'content': self.role},
            {'role': 'user', 'content': prompt}
        ]

    def _apply_instruct_chat_template(self, prompt: str) -> str:
        """Format ``prompt`` into one that conforms to the instruct syntax."""
        tokenizer: PreTrainedTokenizer = self.resource.tokenizer
        return tokenizer.apply_chat_template(
            conversation=self._apply_messages(prompt),
            tokenize=False,
            return_dict=False,
            **self.chat_template_args)

    def _create_template(self, template: Union[str, Path]) -> Template:
        with openread(template, interpret_str=True) as f:
            content: str = f.read()
        env = Environment(loader=BaseLoader, keep_trailing_newline=True)
        return env.from_string(content)

    def _prepare_request(self, request: InstructTaskRequest) -> TaskRequest:
        """Return the input given to the model using the request."""
        if request.model_input is not None:
            logger.debug('request model_input already populated--skipping')
        else:
            template: Template = self._create_template(self.inference_template)
            prompt: str = template.render(request=request, task=self)
            if self.apply_chat_template:
                try:
                    prompt = self._apply_instruct_chat_template(prompt)
                except Exception as e:
                    raise TaskError(
                        f'Could not format prompt using {self.resource}: {e}') \
                        from e
            request.model_input = prompt
        return request

    def _prepare_dataset(self, ds: Dataset, factory: TaskDatasetFactory) -> \
            Dataset:
        def map_batch(batch: LazyBatch) -> Dict[str, List[Dict[str, Any]]]:
            texts: List[Union[str, List[Dict[str, str]]]] = []
            for data in zip(*tuple(map(lambda k: batch[k], keys))):
                params: Dict[str, Any] = dict(zip(keys, data))
                params['task'] = self
                prompt: str = template.render(**params)
                if self.train_apply_chat_template:
                    texts.append(self._apply_instruct_chat_template(prompt))
                else:
                    texts.append(self._apply_messages(prompt))
            return {field: texts}

        field: str = factory.text_field if self.train_apply_chat_template \
            else factory.messages_field
        template: Template = self._create_template(self.train_template)
        keys: Tuple[str, ...] = tuple(ds.features.keys())
        return ds.map(map_batch, batched=True)

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        dct = self.asdict()
        dct.pop('inference_template')
        dct.pop('train_template')
        self._write_object(dct, depth, writer)
        if self.inference_template is not None:
            with openread(self.inference_template, interpret_str=True) as f:
                self._write_line('inference_template:', depth, writer)
                self._write_block(f.read(), depth + 1, writer)
        if self.train_template is not None:
            self._write_line('train_template:', depth, writer)
            with openread(self.train_template, interpret_str=True) as f:
                self._write_block(f.read(), depth + 1, writer)
