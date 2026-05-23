#!/usr/bin/env python

from zensols.cli import ConfigurationImporterCliHarness


def create_harness(args: str = None) -> ConfigurationImporterCliHarness:
    args: str = 'proto' + ('' if args is None else f' {args}')
    return ConfigurationImporterCliHarness(
        app_factory_class='zensols.lmtask.ApplicationFactory',
        proto_args=args,
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
    }[2]
    args: str = f'-c trainconf/{task}-{model}.yml'
    harness: ConfigurationImporterCliHarness = create_harness(args)
    harness.run()


if (__name__ == '__main__'):
    run()
