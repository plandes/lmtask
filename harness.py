#!/usr/bin/env python

from zensols.cli import ConfigurationImporterCliHarness


def create_harness(task: str, model: str) -> ConfigurationImporterCliHarness:
    return ConfigurationImporterCliHarness(
        app_factory_class='zensols.lmtask.ApplicationFactory',
        proto_args=f'-c trainconf/{task}-{model}.yml',
        proto_factory_kwargs={'reload_pattern': r'^zensols.lmtask.(?!task)'})


def run():
    ConfigurationImporterCliHarness.add_sys_path('src')
    from zensols.lmtask.torchconfig import TorchConfig
    TorchConfig.set_random_seed()
    task: str = {
        0: 'tinystory',
        1: 'imdb',
    }[1]
    model: str = {
        0: 'llama3',
        1: 'qwen3',
        2: 'gemma4',
    }[0]
    harness: ConfigurationImporterCliHarness = create_harness(task, model)
    harness.run()


def run_test():
    import logging
    from pathlib import Path
    from zensols.introspect.tester import UnitTester
    logging.basicConfig()
    logging.getLogger('tester').setLevel(logging.INFO)
    ConfigurationImporterCliHarness.add_sys_path('src')
    testrun = UnitTester('test_trained', Path('tests'))
    testrun()


if (__name__ == '__main__'):
    #run()
    run_test()
