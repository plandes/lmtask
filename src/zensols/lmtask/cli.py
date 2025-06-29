"""Command line entry point to the application.

"""
__author__ = 'Paul Landes'

from typing import List, Dict, Any, Type
import sys
from zensols.cli import ActionResult, CliHarness
from zensols.cli import ApplicationFactory as CliApplicationFactory
from .generate import TextGenerator
from . import TaskFactory


class ApplicationFactory(CliApplicationFactory):
    def __init__(self, *args, **kwargs):
        kwargs['package_resource'] = 'zensols.lmtask'
        super().__init__(*args, **kwargs)

    @classmethod
    def get_application(cls: Type) -> TextGenerator:
        """Get a text generator instance."""
        return cls.create_harness().get_application()

    @classmethod
    def get_task_factory(cls: Type) -> TaskFactory:
        """Get the factory that creates tasks."""
        return cls.get_application().task_factory


def main(args: List[str] = sys.argv, **kwargs: Dict[str, Any]) -> ActionResult:
    harness: CliHarness = ApplicationFactory.create_harness(relocate=False)
    harness.invoke(args, **kwargs)
