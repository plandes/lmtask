#!/usr/bin/env python

from zensols.cli import ConfigurationImporterCliHarness


def create_harness(args: str = None) -> ConfigurationImporterCliHarness:
    args: str = 'proto' + ('' if args is None else f' {args}')
    return ConfigurationImporterCliHarness(
        #src_dir_name='src',
        app_factory_class='zensols.lmtask.ApplicationFactory',
        proto_args=args,
        proto_factory_kwargs={'reload_pattern': r'^zensols.lmtask.'})


def run():
    ConfigurationImporterCliHarness.add_sys_path('src')
    from zensols.lmtask.torchconfig import TorchConfig
    TorchConfig.set_random_seed()
    harness: ConfigurationImporterCliHarness = create_harness('-c trainconf/tinystory.yml')
    harness.run()


if (__name__ == '__main__'):
    run()
