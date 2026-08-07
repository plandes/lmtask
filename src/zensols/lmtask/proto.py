"""Prototyping.

"""
from dataclasses import dataclass, field
import logging
from zensols.config import ConfigFactory
from .instruct import InstructTaskRequest
from . import Task, Application
from .app import _Format
logger = logging.getLogger(__name__)


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
        res.write(include_model_output_raw=True)

    def _example_stream_base(self):
        from . import Task
        task: Task = self.app.task_factory.create('base_generate')
        task.generator.generate_params['max_new_tokens'] = 300
        task.generator.stream(self.prompt)

    def _example_stream_instruct(self):
        from . import Task
        task: Task = self.app.task_factory.create('instruct_generate')
        prompt: str = f'Write 20 word short story starting witih "{self.prompt}".'
        task.generator.stream(prompt)

    def _example_prompt_population(self, run: bool = 1):
        from .task import TaskResponse
        task: Task = self.app.task_factory.create('sentiment')
        instr: str = 'I love football.\nI hate olives.\nEarth is big.'
        req = InstructTaskRequest(instruction=instr)
        req = task.prepare_request(req)
        req.write()
        if run:
            res: TaskResponse = task.process(req)
            res.write(include_model_output=True)

    def _example_tiny_story(self):
        """Needs ``proto_args='proto -c trainconf/tinystory.yml'`` in the
        harness args.

        """
        from . import Task
        task: Task = self.app.task_factory.create('dataset')
        task.generator.generate_params['max_new_tokens'] = 300
        task.generator.stream(self.prompt)

    def _example_imdb_review(self):
        """Read reviews (newline separated) and judge sentiment using trained
        IMDB model.

        """
        from .task import TaskResponse
        from pathlib import Path
        task: Task = self.app.task_factory.create('dataset')
        reviews: list[str] = Path('~/review.txt').expanduser().\
            read_text().strip().split('\n')
        for instr in reviews:
            print(instr)
            req = InstructTaskRequest(instruction=instr)
            req = task.prepare_request(req)
            res: TaskResponse = task.process(req)
            res.write(include_model_output=True)
            print('_' * 80)

    def _dump_train(self, name: str):
        from datasets import Dataset
        from .task import TaskDatasetFactory
        from .train import Trainer
        trainer: Trainer = self.app.trainer
        dsf: TaskDatasetFactory = trainer.train_source
        ds: Dataset = dsf.create()
        with open(f'{name}.txt', 'w') as f:
            for row in ds:
                print(row.keys())
                print(row['text'], file=f)
                print('_' * 40, file=f)

    def _tmp(self):
        if 1:
            from .app import _Format
            self.app.test(output_format=_Format.csv)
        else:
            self.app.benchmark()

    def proto(self, run: int = 0):
        {
            0: self._tmp,
            1: self.app.dataset_sample,
            2: self.app.show_task,
            3: lambda: self.app.instruct(
                task_name='instruct_generate',
                instruction='Write a poem about a cat in 50 words or less.',
                output_format=_Format.full),
            4: lambda: self.app.instruct(
                task_name='sentiment',
                instruction='I love football.\nI hate olives.\nEarth is big.',
                output_format=_Format.full),
            5: lambda: self.app.instruct(
                task_name='ner',
                instruction='Obama was the 44th president of the United States.',
                output_format=_Format.full),
            6: self._example_direct_model,
            7: self._example_base_generate,
            8: self._example_stream_base,
            9: self._example_stream_instruct,
            10: self._example_prompt_population,
            11: self._example_tiny_story,
            12: self._example_imdb_review,
        }[run]()
