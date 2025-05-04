"""HuggingFace trainer wrapper.

"""

from typing import Any, Dict, Tuple
from dataclasses import dataclass, field
import logging
from datasets import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.trainer_utils import TrainOutput
import peft
from peft import PeftModelForCausalLM
import trl
from .generate import GeneratorResource
from .train import Trainer, TrainerResource

logger = logging.getLogger(__name__)


@dataclass
class HFTrainerResource(TrainerResource):
    generator_resource: GeneratorResource = field(default=None)
    """The resource used to the source checkpoint."""

    peft_config: peft.LoraConfig = field(default=None)

    def _create_model_tokenizer(self) -> \
            Tuple[PreTrainedTokenizer, PreTrainedModel]:
        res: GeneratorResource = self.generator_resource
        return res.model, res.tokenizer

    def _create_peft_model(self) -> PeftModelForCausalLM:
        model: PreTrainedModel = self.model
        model = peft.prepare_model_for_kbit_training(model)
        model = peft.get_peft_model(model, self.peft_config)
        return model


@dataclass
class HuggingFaceTrainer(Trainer):
    """The HuggingFace trainer."""
    def _train(self, params: Dict[str, Any], ds: Dataset) -> TrainOutput:
        model: PeftModelForCausalLM = self.resource.peft_model
        tokenizer: PreTrainedTokenizer = self.resource.tokenizer
        trainer = trl.SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=ds,
            **params)
        out: TrainOutput = trainer.train()
        return out
