import logging
import warnings
import unittest
from pathlib import Path
import shutil
import os
from zensols.util import Failure
from zensols.config import ConfigFactory
from zensols.cli import CliHarness
from zensols.lmtask import Task, TaskFactory, Application, ApplicationFactory


if 0:
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)


class TestBase(unittest.TestCase):
    def setUp(self):
        self.target = Path('target')
        if self.target.is_dir():
            shutil.rmtree(self.target)

    def _should_protect_hf(self) -> bool:
        return os.environ.get('PROTECT_HF_ACCESS', 'no') == '1'

    def _trained_model_exists(self, task: str, model: str) -> bool:
        train_dir: Path = Path(f'data/{task}/{model}/model/peft')
        return train_dir.is_dir()

    def _get_harness(self) -> CliHarness:
        return ApplicationFactory.create_harness()

    def _get_config_factory(self, task: str = None, model: str = None) -> \
            ConfigFactory:
        harn: CliHarness = self._get_harness()
        args: str = '-c test-resources/test.yml'
        if task is not None:
            assert self._trained_model_exists(task, model)
            args = f'-c trainconf/{task}-{model}.yml'
        args += ' --level=err'
        return harn.get_config_factory(args)

    def _get_application(self) -> Application:
        harn: CliHarness = self._get_harness()
        app: Application = harn.get_application(
            '-c test-resources/test.yml --level=err')
        if isinstance(app, Failure):
            app.rethrow()
        return app

    def _get_trained_task(self, task_name: str, model: str) -> Task:
        if not self._trained_model_exists(task_name, model):
            warnings.warn(
                f"Trained model '{task_name}-{model}' does not exist--skipping",
                UserWarning)
            return
        fac: ConfigFactory = self._get_config_factory(task_name, model)
        task_factory: TaskFactory = fac('lmtask_task_factory')
        #task_factory.write(short=True)
        return task_factory.create('dataset')
