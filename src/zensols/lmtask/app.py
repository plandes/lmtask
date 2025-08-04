"""Large langauage model experimentation.

"""
__author__ = 'Paul Landes'

from dataclasses import dataclass, field
from enum import Enum, auto
import logging
from pathlib import Path
import json
import yaml
from datasets import Dataset
from zensols.config import ConfigFactory
from zensols.cli import ApplicationError
from .instruct import InstructTaskRequest
from . import TaskResponse, JSONTaskResponse, Task, TaskFactory

logger = logging.getLogger(__name__)


class _Format(Enum):
    full = auto()
    text = auto()
    json = auto()
    yaml = auto()


@dataclass
class Application(object):
    """Large langauage model experimentation.

    """
    config_factory: ConfigFactory = field()
    """Used to create training resources."""

    task_factory: TaskFactory = field()
    """Create tasks used to fullfill CLI requests."""

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
        {
            _Format.full: res.write,
            _Format.text: write_text,
            _Format.json: write_json,
            _Format.yaml: write_yaml,
        }[output_format]()

    def _get_trainer(self):
        """Print a sample of the configured (``--config``) dataset."""
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

    def dataset_sample(self, max_sample: int = 1):
        """Print sample(s) of the configured (``--config``) dataset.

        :param max_sample: the number of sample to print

        """
        from pprint import pprint
        import itertools as it
        from . import TaskDatasetFactory
        from .train import Trainer
        trainer: Trainer = self._get_trainer()
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
        trainer: Trainer = self._get_trainer()
        trainer.write(include_training_arguments=long_output)

    def train(self):
        """Train a new model on a configured (``--config``) dataset."""
        from .train import Trainer, ModelResult
        import pickle
        trainer: Trainer = self._get_trainer()
        trainer.write(include_training_arguments=True)
        print('_' * 79)
        result: ModelResult = trainer.train()
        result_path: Path = result.output_dir / 'model-result.dat'
        with open(result_path, 'wb') as f:
            pickle.dump(result, f)
        logger.info(f'wrote: {result_path}')


@dataclass
class PrototypeApplication(object):
    """Used by the Python REPL for prototyping.

    """
    CLI_META = {'is_usage_visible': False}

    config_factory: ConfigFactory = field()
    app: Application = field()
    prompt: str = field(default='Once upon a time, in a galaxy, far far away,')

    def _example_direct_model(self):
        from transformers import PreTrainedTokenizer, PreTrainedModel
        from .generate import GeneratorResource
        from . import Task, TextGenerator
        task: Task = self.app.task_factory.create('base_generate')
        generator: TextGenerator = task.generator
        res: GeneratorResource = generator.resource
        tokenizer: PreTrainedTokenizer = res.tokenizer
        model: PreTrainedModel = res.model
        inputs = tokenizer(self.prompt, return_tensors='pt')
        out = model.generate(inputs.input_ids.to('cuda'), max_length=256)
        print(tokenizer.decode(out[0], skip_special_tokens=True))

    def _example_base_generate(self):
        from . import Task, TaskRequest
        task: Task = self.app.task_factory.create('base_generate')
        req = TaskRequest(self.prompt)
        req.write()
        res = task.process(req)
        res.write(include_model_raw=True)

    def _example_stream_base(self):
        from . import Task
        task: Task = self.app.task_factory.create('base_generate')
        task.generator.stream(self.prompt)

    def _example_stream_instruct(self):
        from . import Task
        task: Task = self.app.task_factory.create('instruct_generate')
        prompt: str = f'Write 20 word short story starting witih "{self.prompt}".'
        task.generator.stream(prompt)

    def _example_prompt_population(self):
        task: Task = self.app.task_factory.create('sentiment')
        req = InstructTaskRequest(
            instruction='I love football.\nI hate olives.\nEarth is big.')
        req = task.prepare_request(req)
        req.write()

    def _example_tiny_story(self):
        # this needs proto_args='proto -c trainconf/tinystory.yml'
        from . import Task
        task: Task = self.app.task_factory.create('tinystory')
        task.generator.generate_params['max_length'] = 500
        task.generator.generate_params['temperature'] = 0.9
        task.generator.stream(self.prompt)

    def _example_databricks_instruct(self):
        # this needs proto_args='proto -c trainconf/dbinstruct.yml'
        from . import Task, InstructTaskRequest
        task: Task = self.app.task_factory.create('instruct_databricks')
        req = InstructTaskRequest(
            instruction=(
                'Provide a detailed explanation of the events and ' +
                'circumstances that led to the outbreak of World War II.'))
        req.context = (
            'The goal is to offer a clear and informative account of the ' +
            'factors, political decisions, and international tensions ' +
            'that played a crucial role in triggering World War II. ' +
            'Ensure that the explanation covers the period leading up ' +
            'to the war, key events, and the involvement of major nations')
        if 0:
            self.config_factory.config['lmtask_task_instruct_databricks'].write()
            task.write()
        else:
            task.generator.generate_params['max_length'] = 500
            res = task.process(req)
            res.write()

    def _example_imdb(self, debug: bool = False):
        # this needs proto_args='proto -c trainconf/dbinstruct.yml'
        import datasets
        from datasets import Dataset
        task: Task = self.app.task_factory.create('imdb')
        ds: Dataset = datasets.load_dataset('stanfordnlp/imdb', split='test')
        ds = ds.shuffle(seed=0)
        ds = ds.select(range(50))
        for review in ds:
            req = InstructTaskRequest(instruction=review['text'])
            if debug:
                req = task.prepare_request(req)
                req.write()
            res: TaskResponse = task.process(req)
            if debug:
                res.write(include_model_output_raw=True)
            should: str = 'positive' if review['label'] == 1 else 'negative'
            pred: str = res.model_output.strip()
            correct: bool = (should == pred)
            print(f'should: {should}, pred: {pred}, correct: {correct}')

    def _tmp(self):
        pass

    def proto(self, run: int = 0):
        {
            0: self._tmp,
            1: self.app.show_task,
            2: lambda: self.app.instruct(
                task_name='instruct_generate',
                instruction='Write a poem about a cat in 50 words or less.',
                output_format=_Format.full),
            3: lambda: self.app.instruct(
                task_name='sentiment',
                instruction='I love football.\nI hate olives.\nEarth is big.',
                output_format=_Format.full),
            4: lambda: self.app.instruct(
                task_name='ner',
                instruction='Obama was the 44th president of the United States.',
                output_format=_Format.full),
            5: self._example_direct_model,
            6: self._example_base_generate,
            7: self._example_stream_base,
            8: self._example_stream_instruct,
            9: self._example_prompt_population,
            10: self._example_tiny_story,
            11: self._example_databricks_instruct,
            12: self._example_imdb,
            13: self.app.dataset_sample,
        }[run]()
