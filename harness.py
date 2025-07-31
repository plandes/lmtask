#!/usr/bin/env python

from zensols.cli import ConfigurationImporterCliHarness


def create_harness(args: str = None) -> ConfigurationImporterCliHarness:
    return ConfigurationImporterCliHarness(
        src_dir_name='src/python',
        app_factory_class='zensols.lmtask.ApplicationFactory',
        proto_args='proto' + ('' if args is None else f' {args}'),
        proto_factory_kwargs={'reload_pattern': r'^zensols.lmtask.'})


def run():
    try:
        from zensols.deeplearn import TorchConfig
        TorchConfig.init()
    except Exception:
        pass
    harness: ConfigurationImporterCliHarness = create_harness('-c trainconf/tinystory.yml')
    harness.run()


if (__name__ == '__main__'):
    run()
