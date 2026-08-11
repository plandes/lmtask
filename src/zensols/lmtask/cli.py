"""Command line entry point to the application.

"""
__author__ = 'Paul Landes'

from typing import Any
import sys
from zensols.cli import ActionResult, CliHarness
from zensols.cli import ApplicationFactory as CliApplicationFactory
from . import TaskFactory


class ApplicationFactory(CliApplicationFactory):
    def __init__(self, *args, **kwargs):
        kwargs['package_resource'] = 'zensols.lmtask'
        super().__init__(*args, **kwargs)

    @classmethod
    def get_application(cls: type, args: str | list[str] = None):
        """Get a text generator instance."""
        return cls.create_harness().get_application(args)

    @classmethod
    def get_task_factory(cls: type) -> TaskFactory:
        """Get the factory that creates tasks."""
        return cls.get_application().task_factory

    @classmethod
    def get_benchmark_runner(cls: type):
        return cls.get_application().benchmark_result


def main(args: list[str] = sys.argv, **kwargs: dict[str, Any]) -> ActionResult:
    harness: CliHarness = ApplicationFactory.create_harness(relocate=False)
    harness.invoke(args, **kwargs)
