"""Prototyping.

"""
from dataclasses import dataclass, field
import logging
from zensols.config import ConfigFactory
from .instruct import InstructTaskRequest
from . import TaskResponse, Task, Application
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

    def _example_prompt_population(self):
        task: Task = self.app.task_factory.create('sentiment')
        req = InstructTaskRequest(
            instruction='I love football.\nI hate olives.\nEarth is big.')
        req = task.prepare_request(req)
        req.write()

    def _example_tiny_story(self):
        """Needs ``proto_args='proto -c trainconf/tinystory.yml'`` in the
        harness args.

        """
        from . import Task
        task: Task = self.app.task_factory.create('tinystory')
        task.generator.generate_params['max_new_tokens'] = 300
        task.generator.stream(self.prompt)

    def _dump_tiny(self):
        from datasets import Dataset
        from . import TaskDatasetFactory
        from .train import Trainer
        trainer: Trainer = self.app._get_trainer()
        dsf: TaskDatasetFactory = trainer.train_source
        ds: Dataset = dsf.create()
        with open('tiny.txt', 'w') as f:
            for row in ds:
                print(row['text'], file=f)
                print('_' * 40, file=f)

    def _example_databricks_instruct(self):
        """Needs ``proto_args='proto -c trainconf/dbinstruct.yml'`` in the
        harness args.

        """
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
        """Needs ``proto_args='proto -c trainconf/dbinstruct.yml'`` in the
        harness args.

        """
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
        self.config_factory.config.write()

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
