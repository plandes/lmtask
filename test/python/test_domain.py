from zensols.util import Failure
from zensols.lmtask import TaskRequest, JSONTaskResponse
from json.decoder import JSONDecodeError
from util import TestBase
from zensols.config import ConfigFactory
from zensols.lmtask import Application
from zensols.lmtask.generate import GeneratorResource


class TestResponse(TestBase):
    def setUp(self):
        self.request = TaskRequest(model_input='some input')
        self.model_output = '["first", 2, {"val1": 1, "val2": 2'
        self.should = ['first', 2, {'val1': 1, 'val2': 2}]

    def _test_partial(self):
        response = JSONTaskResponse(
            request=self.request,
            model_output=self.model_output)
        self.assertEqual(self.should, response.model_output_json)

    def test_robust_error(self):
        response = JSONTaskResponse(
            request=self.request,
            model_output_raw=None,
            model_output=self.model_output,
            robust_json=True)
        res = response.model_output_json
        self.assertEqual(list, type(res))
        self.assertEqual(self.should[0], res[0])

        response = JSONTaskResponse(
            request=self.request,
            model_output_raw=None,
            model_output='{}, {}',
            robust_json=True)
        fail = response.model_output_json
        self.assertTrue(isinstance(fail, Failure))
        self.assertEqual(JSONDecodeError, type(fail.exception))
        self.assertEqual('Could not JSON parse <{}, {}>', fail.message)

    def test_error(self):
        from json.decoder import JSONDecodeError
        response = JSONTaskResponse(
            request=self.request,
            model_output_raw=None,
            model_output='{}, {}',
            robust_json=False)
        with self.assertRaisesRegex(JSONDecodeError, r'^Extra data: line 1'):
            response.model_output_json


class TestFactory(TestBase):
    def setUp(self):
        self.app: Application = self._get_application()

    def test_task_list(self):
        fac = self.app.task_factory
        self.assertEqual(set('poem echo broken_record'.split()), fac.task_names)
        self.assertTrue('poem' in fac)
        self.assertFalse('<NADA>' in fac)

    def test_create(self):
        fac = self.app.task_factory
        task = fac.create('poem')
        self.assertEqual('poem', task.name)
        self.assertEqual('writes nice poems', task.description)


class TestGeneratorResource(TestBase):
    def setUp(self):
        self.app: Application = self._get_application()

    def test_model_desc_set(self):
        fac: ConfigFactory = self._get_config_factory()
        res: GeneratorResource = fac('lmtask_llama_test_desc_set_resource')
        self.assertEqual('Llama-3.1-8B-Instruct', res.model_desc)
        self.assertEqual('llama-3-1-8b-instruct', res.model_file_name)

    def test_model_desc_not_set_model_id(self):
        fac: ConfigFactory = self._get_config_factory()
        res: GeneratorResource = fac(
            'lmtask_llama_test_desc_not_set_model_id_resource')
        self.assertEqual('not_set_with_user', res.model_desc)
        self.assertEqual('not-set-with-user', res.model_file_name)

    def test_model_desc_not_set_checkpoint(self):
        fac: ConfigFactory = self._get_config_factory()
        res: GeneratorResource = fac(
            'lmtask_llama_test_desc_not_set_checkpoint_resource')
        self.assertEqual('checkpoint-123', res.model_desc)
        self.assertEqual('checkpoint-123', res.model_file_name)
