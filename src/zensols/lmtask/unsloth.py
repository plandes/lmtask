"""Unsloth trainer wrapper.

"""
__author__ = 'Paul Landes'

from typing import Any, Dict, Tuple
from dataclasses import dataclass, field
import logging
from datasets import Dataset
from peft import PeftModelForCausalLM
from transformers import PreTrainedTokenizer, PreTrainedModel
from transformers.trainer_utils import TrainOutput
import unsloth
from unsloth import FastLanguageModel
from .train import Trainer, TrainerResource
from .generate import GeneratorResource

logger = logging.getLogger(__name__)


@dataclass
class UnslothTrainerResource(TrainerResource):
    generator_resource: GeneratorResource = field(default=None)
    """The resource used to the source checkpoint."""

    peft_args: Dict[str, Any] = field(default=None)
    """The parameters used to create the PEFT model."""

    def _create_model_tokenizer(self) -> \
            Tuple[PreTrainedTokenizer, PreTrainedModel]:
        model, tok = FastLanguageModel.from_pretrained(**self.model_args)
        self.generator_resource.configure_model(model)
        self.generator_resource.configure_tokenizer(tok)
        return model, tok

    def _create_peft_model(self) -> PeftModelForCausalLM:
        return FastLanguageModel.get_peft_model(self.model, **self.peft_args)


@dataclass
class UnslothTrainer(Trainer):
    """Uses the Unsloth API for faster training."""
    def _train(self, params: Dict[str, Any], ds: Dataset) -> TrainOutput:
        logger.info('unsloth might complain about an unset padding token')
        logger.info('this was fixed after the tokenizer construction')
        model: PeftModelForCausalLM = self.resource.peft_model
        tokenizer: PreTrainedTokenizer = self.resource.tokenizer
        trainer = unsloth.UnslothTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=ds,
            **params)
        return trainer.train()
