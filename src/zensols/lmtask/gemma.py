"""Temporary fix to get Gemma 4 to fine-tune with LoRA.

"""
from dataclasses import dataclass
from torch import nn
from transformers import PreTrainedModel
from .hf import HFTrainerResource


@dataclass
class Gemma4HFTrainerResource(HFTrainerResource):
    def unwrap_gemma4_clippable_linear(self, module: nn.Module):
        for name, child in list(module.named_children()):
            if child.__class__.__name__ == "Gemma4ClippableLinear" and \
               hasattr(child, "linear"):
                setattr(module, name, child.linear)
            else:
                self.unwrap_gemma4_clippable_linear(child)

    def _configure_model(self, model: PreTrainedModel) -> PreTrainedModel:
        self.unwrap_gemma4_clippable_linear(model)
        return model
