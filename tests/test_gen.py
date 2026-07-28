from zensols.lmtask import Task, TaskRequest, TaskResponse
from zensols.lmtask.torchconfig import TorchConfig
from util import TestBase


class TestGenerate(TestBase):
    DEBUG: bool = 0
    PROMPT: str = 'Once upon a time, in a galaxy, far far away,'
    STREAM: bool = 0
    DETERMINISTIC: bool = 1
    MIN_WORDS: int = 20

    def setUp(self):
        super().setUp()
        TorchConfig.set_random_seed()

    def _test_generate(self, task_name: str, model: str, task: str = 'dataset',
                       clear: bool = False, assert_period: bool = True):
        task: Task = self._get_trained_task(task_name, model, task)
        if task is None:
            return
        if clear:
            task.generator.resource.clear()
        if task is None:
            return
        req = TaskRequest(self.PROMPT)
        task.generator.generate_params['max_new_tokens'] = 300
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
        if assert_period:
            self.assertTrue(
                out.endswith('.'),
                f'expected output to end with a period: <<{out}>>')

    def test_default_llama3(self):
        self._test_generate(None, 'llama3', 'base_generate', True, False)

    def test_tinystory_llama3(self):
        self._test_generate('tinystory', 'llama3', 'dataset', True)

    def test_tinystory_qwen3(self):
        self._test_generate('tinystory', 'qwen3')

    def test_tinystory_gemma4(self):
        self._test_generate('tinystory', 'gemma4')

    def test_tinystory_dsr1qwen(self):
        self._test_generate('tinystory', 'dsr1qwen3')
