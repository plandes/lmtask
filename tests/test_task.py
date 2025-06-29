import warnings
import torch
from zensols.lmtask.instruct import InstructTaskRequest
from zensols.lmtask import (
    TaskError, Task, TaskRequest, TaskResponse, Application
)
from util import TestBase


class TestTask(TestBase):
    def setUp(self):
        super().setUp()
        self.app: Application = self._get_application()
        self.cache_file = self.target / 'poem.sqlite3'

    def test_task_api(self):
        fac = self.app.task_factory
        task: Task = fac.create('broken_record')
        request = TaskRequest(model_input='test')
        response: TaskResponse = task.process(request)
        should: str = "it's 12 noon"
        self.assertEqual(should, response.model_output_raw)
        self.assertEqual(should, response.model_output)
        self.assertEqual(request, response.request)

    def test_prompt_format(self):
        fac = self.app.task_factory
        task: Task = fac.create('echo')
        request = InstructTaskRequest(instruction='inst')
        response: TaskResponse = task.process(request)
        mi: str = response.request.model_input
        should: str = '<|begin_of_text|><|start_header_id|>system<|end_header_id|>'
        self.assertTrue(mi.startswith(should), f'prompt: <<{mi}>>')
        should = 'instruct: inst<|eot_id|>'
        self.assertTrue(mi.endswith(should), f'prompt: <<{mi}>>')
        should = "it's 12 noon"
        self.assertEqual(should, response.model_output_raw)
        self.assertEqual(should, response.model_output)
        self.assertEqual(request, response.request)

    def test_prompt_format_raw(self):
        fac = self.app.task_factory
        task: Task = fac.create('echo')
        raw_text: str = 'raw input'
        request = InstructTaskRequest(model_input=raw_text)
        response: TaskResponse = task.process(request)
        mi: str = response.request.model_input
        should: str = raw_text
        self.assertEqual(raw_text, mi)
        should = "it's 12 noon"
        self.assertEqual(should, response.model_output_raw)
        self.assertEqual(should, response.model_output)
        self.assertEqual(request, response.request)

    def test_assert_request_type(self):
        fac = self.app.task_factory
        task: Task = fac.create('echo')
        request = TaskRequest(model_input='test')
        should: str = r'^Expecting request type .*Instruct'
        with self.assertRaisesRegex(TaskError, should):
            task.process(request)

    def _test_generate(self):
        fac = self.app.task_factory
        task: Task = fac.create('poem')
        instruction: str = 'Write a short poem about a a fuzzy bear.'
        request = InstructTaskRequest(instruction=instruction)
        self.assertFalse(self.cache_file.exists())
        response = task.process(request)
        self.assertTrue(self.cache_file.exists())
        self.assertEqual(request, response.request)
        self.assertEqual(id(request), id(response.request))
        self.assertEqual(instruction, response.request.instruction)
        self.assertFalse(response.model_output is None)
        self.assertEqual(str, type(response.model_output))
        self.assertTrue(len(response.model_output) > 50)
        self.assertTrue(len(response.model_output.split('\n')) > 3)

    def test_generate(self):
        if torch.cuda.is_available():
            self._test_generate()
        else:
            warnings.warn('No GPU available--skipping task generate test.',
                          UserWarning)
