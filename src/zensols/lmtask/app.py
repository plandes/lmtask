"""Task-specialized language model training and inference.

"""
__author__ = 'Paul Landes'
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
import logging
from pathlib import Path
import json
import yaml
from zensols.config import ConfigFactory
from zensols.util.std import stdout
from zensols.cli import ApplicationError
from .instruct import InstructTaskRequest
from . import TaskResponse, JSONTaskResponse, Task, TaskFactory

logger = logging.getLogger(__name__)


class _Format(Enum):
    full = auto()
    text = auto()
    json = auto()
    yaml = auto()
    csv = auto()


@dataclass
class Application(object):
    """Task-specialized language model training and inference.

    """
    config_factory: ConfigFactory = field()
    """Used to create configured application and training resources."""

    task_factory: TaskFactory = field()
    """Create tasks used to fulfill CLI requests."""

    def _get_task(self, task_name: str) -> Task:
        if task_name not in self.task_factory:
            raise ApplicationError(f"No such task available: {task_name}")
        return self.task_factory.create(task_name)

    def show_task(self, task_name: str = None):
        """Print the configuration of a task if ``--name`` is given, otherise a
        list of available tasks.

        :param task_name: the task that creates the prompt and parses the result

        """
        if task_name is None:
            self.task_factory.write(short=True)
        else:
            task: Task = self._get_task(task_name)
            task.write()

    def stream(self, task_name: str, prompt: str):
        """Stream generated text from the model.

        :param task_name: the task that generates the result

        :param prompt: the prompt text as input to the model

        """
        task: Task = self._get_task(task_name)
        task.generator.stream(prompt)

    def instruct(self, task_name: str, instruction: str, role: str = None,
                 output_format: _Format = None):
        """Generate text by inferencing with the model.

        :param task_name: the task that generates the result

        :param instruction: added to the prompt to instruction the model

        :param role: the role the model takes

        :param output_format: data format for the output

        """
        def write_text():
            for text in res.model_output:
                print(text)

        def write_json():
            if not isinstance(res, JSONTaskResponse):
                raise ApplicationError(
                    f"Task '{task_name}' has no JSON response")
            print(json.dumps(res.model_output_json, indent=4))

        def write_yaml():
            if not isinstance(res, JSONTaskResponse):
                raise ApplicationError(
                    f"Task '{task_name}' has no JSON response")
            output: str = yaml.dump(
                data=list(res.model_output_json),
                default_flow_style=False)
            output = output.rstrip()
            print(output)

        task: Task = self._get_task(task_name)
        if role is not None:
            task.role = role
        output_format = _Format.full if output_format is None else output_format
        req = InstructTaskRequest(instruction=instruction)
        res: TaskResponse = task.process(req)
        fn: Callable = {
            _Format.full: res.write,
            _Format.text: write_text,
            _Format.json: write_json,
            _Format.yaml: write_yaml,
        }.get(output_format)
        if fn is None:
            raise ApplicationError(f'Format {output_format} is not supported')
        fn()

    @property
    def trainer(self) -> 'Trainer':
        """The currently configured :class:`.train.Trainer`."""
        from zensols.config import Settings
        from .train import Trainer
        def_sec: str = 'lmtask_trainer_default'
        def_property: str = 'trainer_name'
        defaults: Settings = self.config_factory(def_sec)
        trainer_name: str = defaults.get(def_property)
        if trainer_name is None:
            raise ApplicationError(
                f"Configuration has no '{def_property}' in section " +
                f"'{def_sec}'; missing --config?")
        trainer: Trainer = self.config_factory(trainer_name)
        if trainer.train_source is None:
            raise ApplicationError(
                f'Configuration did not set train source on {trainer_name}')
        return trainer

    @property
    def tester(self) -> 'Tester':
        """The currently configured :class:`.test.Tester`."""
        from .test import Tester
        tester: Tester = self.config_factory('lmtask_tester')
        return tester

    def dataset_sample(self, max_sample: int = 1):
        """Print sample(s) of the configured (``--config``) dataset.

        :param max_sample: the number of sample to print

        """
        from pprint import pprint
        import itertools as it
        from datasets import Dataset
        from . import TaskDatasetFactory
        from .train import Trainer
        trainer: Trainer = self.trainer
        dsf: TaskDatasetFactory = trainer.train_source
        ds: Dataset = dsf.create()
        for row in it.islice(ds, max_sample):
            print('_' * 40)
            pprint(row)

    def show_trainer(self, long_output: bool = False):
        """Print configuration and dataset stats of the configured
        (``--config``) trainer.

        :param long_output: verbosity

        """
        from .train import Trainer
        trainer: Trainer = self.trainer
        trainer.write(include_training_arguments=long_output)

    def train(self):
        """Train a new model on a configured (``--config``) dataset."""
        from .train import Trainer, TrainResult
        trainer: Trainer = self.trainer
        trainer.write(include_training_arguments=True)
        print('_' * 79)
        result: TrainResult = trainer.train()
        trainer.save_result(result)

    def test(self, output_file: Path = Path('-'),
             output_format: _Format = None):
        """Test a trained model on a configured (``--config``) dataset.

        :param output_file: output file name, ``-`` for standard out

        :param output_format: data format for the output

        """
        from .test import Tester, TestResult
        output_format = _Format.json if output_format is None else output_format
        tester: Tester = self.config_factory('lmtask_tester')
        res: TestResult
        if tester.result_exists:
            res = tester.load_result()
        else:
            res = tester.test()
        with stdout(output_file, extension=output_format.name,
                    logger=logger) as f:
            fn: Callable = {
                _Format.json: lambda: res.write_jsonl(writer=f),
                _Format.csv: lambda: res.dataframe.to_csv(f, index=False),
            }.get(output_format)
            if fn is None:
                raise ApplicationError(
                    f'Format {output_format} is not supported')
            fn()

    def benchmark(self):
        """Test the model and output benchmark files."""
        from .benchmark import BenchmarkRunner
        bench: BenchmarkRunner = self.config_factory('lmtask_benchmark')
        bench.run()
