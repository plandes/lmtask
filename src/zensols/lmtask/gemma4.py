"""Temporary fix to get Gemma 4 to fine-tune with LoRA.

"""
from dataclasses import dataclass
from torch import nn
from transformers import PreTrainedModel, PreTrainedTokenizer
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
        self._config_model_tokens(model)

    def _config_model_tokens(self, model: PreTrainedModel):
        tokenizer: PreTrainedTokenizer = self.tokenizer
        # for Gemma4Config / wrapper configs
        if hasattr(model.config, 'text_config'):
            model.config.text_config.bos_token_id = tokenizer.bos_token_id
            model.config.text_config.eos_token_id = tokenizer.eos_token_id
            model.config.text_config.pad_token_id = tokenizer.pad_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.bos_token_id = tokenizer.bos_token_id
        model.config.eos_token_id = tokenizer.eos_token_id
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id
        model.generation_config.bos_token_id = tokenizer.bos_token_id


@dataclass
class Gemma4HFTrainerResource(HFTrainerResource):
    """
    :see: :class:`.Gemma4GeneratorResource`
    """
    def _configure_model(self, model: PreTrainedModel):
        _unwrap_gemma4_clippable_linear(model)
