#!/usr/bin/env python

from zensols.cli import ConfigurationImporterCliHarness


def create_harness(args: str = None) -> ConfigurationImporterCliHarness:
    args: str = 'proto' + ('' if args is None else f' {args}')
    return ConfigurationImporterCliHarness(
        src_dir_name='src',
        app_factory_class='zensols.lmtask.ApplicationFactory',
        proto_args=args,
        proto_factory_kwargs={'reload_pattern': r'^zensols.lmtask.'})


def set_random_seed(seed: int = 0, disable_cudnn: bool = True,
                    rng_state: bool = True):
    """Set the random number generator for PyTorch (taken from
    :meth:`~zensols.deeplearn.TorchConfig.set_random_seed`).


    :param seed: the random seed to be set

    :param disable_cudnn: if ``True`` disable NVidia's backend cuDNN
                          hardware acceleration, which might have
                          non-deterministic features

    :param rng_state: set the CUDA random state array to zeros

    :see: `Torch Random Seed <https://discuss.pytorch.org/t/random-seed-initialization/7854>`_

    :see: `Reproducibility <https://discuss.pytorch.org/t/non-reproducible-result-with-gpu/1831>`_

    """
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        if rng_state:
            new_states = []
            for state in torch.cuda.get_rng_state_all():
                zeros = torch.zeros(state.shape, dtype=state.dtype)
                new_states.append(zeros)
            torch.cuda.set_rng_state_all(new_states)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(0)

    if disable_cudnn:
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def run():
    set_random_seed()
    harness: ConfigurationImporterCliHarness = create_harness('-c trainconf/tinystory.yml')
    harness.run()


if (__name__ == '__main__'):
    run()
