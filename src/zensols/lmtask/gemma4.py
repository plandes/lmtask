"""Temporary fix to get Gemma 4 to fine-tune with LoRA.

"""
from dataclasses import dataclass
from torch import nn
from transformers import PreTrainedModel
from peft import PeftModel
from .generate import GeneratorResource
from .hf import HFTrainerResource


def _unwrap_gemma4_clippable_linear(module: nn.Module):
    """
    :see: :class:`.Gemma4GeneratorResource`
    """
    for name, child in list(module.named_children()):
        if child.__class__.__name__ == "Gemma4ClippableLinear" and \
           hasattr(child, "linear"):
            setattr(module, name, child.linear)
        else:
            _unwrap_gemma4_clippable_linear(child)


@dataclass
class Gemma4GeneratorResource(GeneratorResource):
    """A class to "monkey patch" the an open issue with using Gemma 4 for text
    only SFT with LoRA.

    :link: https://github.com/huggingface/peft/issues/3129

    :link: https://huggingface.co/google/gemma-4-31B/discussions/3

    """
    def _configure_peft(self, model: PeftModel):
        for attr in 'temperature top_p top_k'.split():
            setattr(model.generation_config, attr, None)
        _unwrap_gemma4_clippable_linear(model)

    def _configure_model(self, model: PreTrainedModel):
        for attr in 'temperature top_p top_k'.split():
            setattr(model.generation_config, attr, None)


@dataclass
class Gemma4HFTrainerResource(HFTrainerResource):
    """
    :see: :class:`.Gemma4GeneratorResource`
    """
    def _configure_model(self, model: PreTrainedModel):
        _unwrap_gemma4_clippable_linear(model)
