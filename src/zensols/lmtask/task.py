"""Task implementations.

"""
from __future__ import annotations
__author__ = 'Paul Landes'
from typing import (
    Tuple, List, Dict, Set, Iterable,
    Optional, Type, Union, ClassVar
)
from dataclasses import dataclass, field
from abc import abstractmethod, ABCMeta
import sys
import logging
import re
import textwrap as tw
from io import TextIOBase, StringIO
from json.decoder import JSONDecodeError
from datasets import Dataset, IterableDataset
from zensols.util import APIError, Failure, WritableContext, Writable
from zensols.config import Dictable, ConfigFactory
from zensols.persist import persisted, PersistableContainer

logger = logging.getLogger(__name__)


class TaskError(APIError):
    """Raised for any LLM specific error in this API.

    """
    def __init__(self, message: str, prompt: str = None):
        if prompt is not None:
            message = f'{message} with prompt: <<{prompt}>>'
        super().__init__(message)
        self.prompt = prompt


@dataclass
class TaskObject(PersistableContainer, Dictable):
    """Base class for task requests and responses.

    """
    def __post_init__(self):
        super().__init__()


@dataclass
class TaskRequest(TaskObject):
    """The input request to the LLM via :meth:`Task.process`.  In most cases,
    obj:`model_input` can be used to skip the prompt compilation step.

    """
    model_input: str = field(default=None)
    """The text given verbatim to the model.  This is some combination of both
    :obj:`querty` and :obj:`prompt`.

    """
    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        self._write_line('model_input:', depth, writer)
        if self.model_input is not None:
            self._write_block(self.model_input, depth + 1, writer)

    def _shorten_value(self, attr: str):
        val: Optional[str] = getattr(self, attr)
        if val is None:
            return f'{attr}: <none>'
        else:
            return f'{attr}: {tw.shorten(val, 40)}'

    def __str__(self) -> str:
        return self._shorten_value('model_input')

    def __repr__(self) -> str:
        return self.__str__()


@dataclass
class TaskResponse(TaskObject):
    """The happy-path response given by :class:`.Task`.

    """
    request: TaskRequest = field()
    """The request used to generated this response."""

    model_output_raw: str = field()
    """The model output verbatim."""

    model_output: str = field()
    """This task instance's parsed response text given by the model."""

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout,
              include_request: bool = False,
              include_model_output: bool = True,
              include_model_output_raw: bool = False):
        if include_request:
            self._write_line('request:', depth, writer)
            self._write_object(self.request, depth + 1, writer)
        if include_model_output:
            self._write_line('model_output:', depth, writer)
            self._write_block(self.model_output, depth + 1, writer)
        if include_model_output_raw:
            self._write_line('raw:', depth, writer)
            self._write_object(self.model_output_raw, depth + 1, writer)


@dataclass
class JSONTaskResponse(TaskResponse):
    """A task that parses the responses as JSON.  The JSON is parsed as much as
    possible and does not raise errors when the json is incomplete.

    """
    _DICTABLE_ATTRIBUTES: ClassVar[Set[str]] = {'model_output_json'}

    robust_json: bool = field(default=True)
    """Whether to return :class:`~zensols.util.fail.Failure` from
    :obj:`model_output_json` instead of raising from parse failures.

    """
    @property
    @persisted('_model_output_json', transient=True)
    def model_output_json(self) -> Union[Failure, str]:
        """The :obj:`response` attribute parsed as JSON.

        :raises json.decoder.JSONDecodeError: if the JSON failed to parse

        :see: obj:`robust_json`

        """
        import partial_json_parser as json
        from partial_json_parser.core.exceptions import MalformedJSON

        mo: str = self.model_output
        if self.robust_json:
            try:
                return json.loads(mo)
            except (JSONDecodeError, MalformedJSON) as e:
                return Failure(
                    exception=e,
                    message=f'Could not JSON parse <{tw.shorten(mo, 70)}>')
        else:
            return json.loads(mo)

    @property
    def any_failures(self) -> bool:
        """Whether any failures were created during JSON parsing."""
        return any(map(lambda x: isinstance(x, Failure),
                       self.model_output_json))

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout,
              include_request: bool = False, include_model_output: bool = False,
              include_json: bool = True):
        if self.any_failures:
            include_model_output = True
        super().write(depth, writer, include_request=include_request,
                      include_model_output=include_model_output)
        if include_json:
            jout: Union[str, Failure] = self.model_output_json
            self._write_line('model_output_json:', depth, writer)
            if isinstance(jout, str):
                self._write_block(jout, depth + 1, writer)
            else:
                self._write_object(jout, depth + 1, writer)


class TaskDatasetFactoryError(TaskError):
    """Raised when :class:`.TaskDatasetFactory` instances can not create
    datasets.

    """
    pass


@dataclass
class TaskDatasetFactory(Dictable, metaclass=ABCMeta):
    """Subclasses create a dataframes used by :class:`.Trainer` and optionally
    does post processing (i.e. filtering and mapping).

    """
    _DICTABLE_WRITABLE_DESCENDANTS = True
    """Force recursive write to use :meth:`task.write`."""

    task: Task = field()
    """The task that helps format text in datasets."""

    text_field: str = field(default='text')
    """The target text field used by the trainer."""

    messages_field: str = field(default='messages')
    """The target conversational field used by the trainer."""

    eval_field: str = field(default='text')
    """The field used for comparison with the the evaluation dataset."""

    def _pre_process(self, ds: Dataset) -> Dataset:
        return ds

    def _post_process(self, ds: Dataset) -> Dataset:
        return ds

    @abstractmethod
    def _create(self) -> Dataset:
        pass

    def _prepare_dataset(self, ds: Dataset) -> Dataset:
        return self.task.prepare_dataset(ds, self)

    def create(self) -> Dataset:
        """Create a new dataset based on :obj:`source`.

        :return: the new dataset after modification by :obj:`post_process`

        """
        ds: Dataset = self._create()
        if self.pre_process:
            ds = self._pre_process(ds)
        ds = self._prepare_dataset(ds)
        if self.post_process:
            ds = self._post_process(ds)
        if not isinstance(ds, (Dataset, IterableDataset)):
            raise TaskDatasetFactoryError(
                f'Expecting Dataset but got: {type(ds)}')
        return ds

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        Writable.write(self, depth, writer)

    def _write(self, c: WritableContext):
        c(self.task, 'task')
        c(self.text_field, 'text_field')


@dataclass
class Task(Dictable, metaclass=ABCMeta):
    """Subclasses turn a prompt and query into a response from an LLM.

    """
    name: str = field()
    """The name of the task."""

    description: str = field()
    """A description of the task."""

    request_class: Type[TaskRequest] = field()
    """The response data."""

    response_class: Type[TaskResponse] = field()
    """The response data."""

    @abstractmethod
    def _process(self, request: TaskRequest) -> TaskResponse:
        """See :meth:`process`."""
        pass

    def _assert_class(self, request: TaskRequest):
        if not isinstance(request, self.request_class):
            raise TaskError(f'Expecting request type {self.request_class}, ' +
                            f'but got: {type(request)}')

    def _prepare_request(self, request: TaskRequest) -> TaskRequest:
        """Optional override."""
        return request

    def _prepare_dataset(self, ds: Dataset, factory: TaskDatasetFactory) -> \
            Dataset:
        """Most will need to override."""
        return ds

    def prepare_request(self, request: TaskRequest) -> TaskRequest:
        """Return a request with the contents populated with a formatted
        prompt.

        """
        self._assert_class(request)
        return self._prepare_request(request)

    def prepare_dataset(self, ds: Dataset, factory: TaskDatasetFactory) -> \
            Dataset:
        """Massage the any data for training necessary to train this task.  This
        might involve apply templates and/or adding terminating tokens.

        """
        return self._prepare_dataset(ds, factory)

    def process(self, request: TaskRequest) -> TaskResponse:
        """Invoke the :obj:`generator` to query the LLM, then return a JSON
        formatted data.

        :param query: a query that is phrased with the assumption that JSON is
                      given as a response

        """
        self._assert_class(request)
        return self._process(request)

    def clear(self):
        """Clear any generator state or cache."""
        pass

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f'{self.name} ({self.description})'


@dataclass
class TaskFactory(Dictable):
    """Creates instances of :class:`.Task` using :meth:`create`.

    """
    config_factory: ConfigFactory = field()
    """The factory that creates tasks."""

    _task_pattern: re.Pattern = field()
    """The pattern used to parse the name of the task from the section text."""

    @property
    @persisted('__name2sec')
    def _name2sec(self) -> Dict[str, str]:
        def map_sec(s: str) -> str:
            m: re.Match = self._task_pattern.match(s)
            if m is not None:
                return m[1], m[0]

        sec_names: Iterable[Tuple[str, str]] = filter(
            lambda n: n is not None,
            map(map_sec, self.config_factory.config.sections))
        return dict(sec_names)

    @property
    @persisted('_task_names')
    def task_names(self) -> Set[str]:
        """The names of the tasks available to create with :meth:`create`."""
        return frozenset(self._name2sec.keys())

    def create(self, name: str) -> Task:
        """Create a new instance of a task with ``name`` per the app config.

        :see: :meth:`task_names`

        """
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'creating task: {name}')
        name2sec: Dict[str, str] = self._name2sec
        sec: str = name2sec.get(name)
        if sec is None:
            raise TaskError(f"No such task: '{name}'")
        task: Task = self.config_factory.new_instance(sec)
        task.name = name
        return task

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout,
              short: bool = False):
        names: List[str] = sorted(self.task_names)
        if short:
            for task_name in names:
                task: Task = self.create(task_name)
                self._write_line(repr(task), depth, writer)
        else:
            max_line_len: int = 0
            wvals: List[str] = []
            task_name: str
            for i, task_name in enumerate(names):
                task: Task = self.create(task_name)
                sio = StringIO()
                task.write(writer=sio)
                wval = sio.getvalue()
                wvals.append(wval)
                max_line_len = max(max_line_len, *map(len, wval.split('\n')))
            for i, (task_name, wval) in enumerate(zip(names, wvals)):
                writer.write(wval)
                if i < len(names) - 1:
                    self._write_divider(depth, writer, width=max_line_len)

    def __contains__(self, name: str) -> bool:
        return name in self.task_names
