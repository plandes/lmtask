import logging
import unittest
from pathlib import Path
import shutil
import os
from zensols.util import Failure
from zensols.config import ConfigFactory
from zensols.cli import CliHarness
from zensols.lmtask import Application, ApplicationFactory


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

    def _get_harness(self) -> CliHarness:
        return ApplicationFactory.create_harness()

    def _get_config_factory(self) -> ConfigFactory:
        harn: CliHarness = self._get_harness()
        return harn.get_config_factory(
            '-c test-resources/test.yml --level=err')

    def _get_application(self) -> Application:
        harn: CliHarness = self._get_harness()
        app: Application = harn.get_application(
            '-c test-resources/test.yml --level=err')
        if isinstance(app, Failure):
            app.rethrow()
        return app
