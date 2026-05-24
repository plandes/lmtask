from zensols.lmtask import Task, TaskRequest, TaskResponse
from zensols.lmtask.torchconfig import TorchConfig
from util import TestBase


class TestTrainedGenerate(TestBase):
    DEBUG: bool = 0
    PROMPT: str = 'Once upon a time, in a galaxy, far far away,'
    STREAM: bool = 0
    DETERMINISTIC: bool = 1
    MIN_WORDS: int = 20

    def setUp(self):
        super().setUp()
        TorchConfig.set_random_seed()

    def _test_gen(self, model: str):
        task_name: str = 'tinystory'
        task: Task = self._get_trained_task(task_name, model)
        req = TaskRequest(self.PROMPT)
        if self.DEBUG:
            print(f'testing: {task_name}-{model}')
        if self.DETERMINISTIC:
            model = task.generator.resource.model
            model.generation_config.do_sample = False
            model.generation_config.temperature = None
            model.generation_config.top_p = None
            model.generation_config.top_k = None
        if self.STREAM:
            task.generator.stream(self.PROMPT)
            return
        if self.DEBUG:
            req.write()
        res: TaskResponse = task.process(req)
        if self.DEBUG:
            res.write(include_model_output_raw=True)
        out: str = res.model_output
        word_len: int = len(out)
        self.assertTrue(
            word_len > self.MIN_WORDS,
            f'expected at least {self.MIN_WORDS} but got {word_len}')
        self.assertTrue(
            out.endswith('.'),
            f'expected output to end with a period: <<{out}>>')

    def test_generate(self):
        model: str
        for model in self.MODELS:
            self._test_gen(model)
