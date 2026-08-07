#!/usr/bin/env python

import os
from zensols.cli import ConfigurationImporterCliHarness


def create_harness(task: str, model: str) -> ConfigurationImporterCliHarness:
    if task is None:
        args = f'resource(zensols.lmtask): resources/models/{model}.conf'
    else:
        args = f'trainconf/{task}-{model}.yml'
    args = f"proto -c '{args}'"
    return ConfigurationImporterCliHarness(
        app_factory_class='zensols.lmtask.ApplicationFactory',
        proto_args=args,
        proto_factory_kwargs={'reload_pattern': r'^zensols.lmtask.(?!task)'})


def run():
    from zensols.lmtask.torchconfig import TorchConfig
    TorchConfig.set_random_seed()
    task: str = {
        0: None,
        1: 'tinystory',
        2: 'imdb',
        3: 'fpb',
    }[3]
    model: str = {
        0: 'llama3',
        1: 'qwen3',
        2: 'gemma4',
        3: 'dsr1qwen3',
    }[2]
    harness: ConfigurationImporterCliHarness = create_harness(task, model)
    harness.run()


def run_test():
    import logging
    from pathlib import Path
    from zensols.introspect.tester import UnitTester
    logging.basicConfig()
    logging.getLogger('tester').setLevel(logging.INFO)
    testrun = UnitTester('test_instruct', Path('tests'))
    testrun()


if (__name__ == '__main__'):
    ConfigurationImporterCliHarness.add_sys_path('src')
    if os.environ.get('HARNESS_TEST', '0') == '1':
        run_test()
    else:
        run()
