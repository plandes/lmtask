"""HuggingFace trainer wrapper.

"""
from typing import Any, Dict, Tuple
from dataclasses import dataclass, field
import logging
from datasets import Dataset
from io import StringIO
from transformers import (
    PreTrainedModel, PreTrainedTokenizer, BitsAndBytesConfig
)
from transformers.trainer_utils import TrainOutput
import peft
from peft import LoraConfig, PeftModelForCausalLM
import trl
from zensols.util import stdwrite
from .generate import GeneratorResource
from .train import Trainer, TrainerResource

logger = logging.getLogger(__name__)


@dataclass
class HFTrainerResource(TrainerResource):
    """Uses :class:`.HuggingFaceTrainer` for training the model.

    """
    generator_resource: GeneratorResource = field(default=None)
    """The resource used to the source checkpoint."""

    peft_config: LoraConfig = field(default=None)
    """The Peft low rank adapters configuration."""

    def _create_model_tokenizer(self) -> \
            Tuple[PreTrainedTokenizer, PreTrainedModel]:
        res: GeneratorResource = self.generator_resource
        return res.model, res.tokenizer

    def _create_peft_model(self) -> PeftModelForCausalLM:
        """Create the Peft model for LoRA training.  The quantization is set
        :obj:`model_args` in the ``quantization_config``, which is called in
        :meth:`load_model`.  The LoRA coniguration is set in :obj:`peft_config`.

        :link: `HF: <https://huggingface.co/docs/peft/en/developer_guides/quantization>`_

        """
        model: PreTrainedModel = self.model
        quant: BitsAndBytesConfig = self.generator_resource.\
            model_args.get('quantization_config')
        if quant is None:
            logger.debug('no quantization configured')
        else:
            assert isinstance(quant, BitsAndBytesConfig)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'configuring for quantization: {quant}')
            # preprocess the quantized model for training
            model = peft.prepare_model_for_kbit_training(model)
        # create a PeftModel from the (optionally) quantized model
        model: PeftModelForCausalLM = peft.get_peft_model(
            model, self.peft_config)
        if logger.isEnabledFor(logging.INFO):
            sio = StringIO()
            with stdwrite(sio):
                model.print_trainable_parameters()
            logger.info(sio.getvalue().strip())
        return model


@dataclass
class HuggingFaceTrainer(Trainer):
    """The HuggingFace trainer.

    """
    def _train(self, params: Dict[str, Any], train_ds: Dataset) -> TrainOutput:
        """Train using :class:`~transformers.sft.SFTTrainer.

        :link: `Quick Tour: <https://huggingface.co/docs/peft/en/quicktour>`_

        :link: `Trainer: <https://huggingface.co/docs/trl/en/sft_trainer>`_

        """
        model: PeftModelForCausalLM = self.resource.peft_model
        tokenizer: PreTrainedTokenizer = self.resource.tokenizer
        trainer = trl.SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=train_ds,
            **params)
        logger.info('starting HuggingFace trainer...')
        out: TrainOutput = trainer.train()
        if logger.isEnabledFor(logging.INFO):
            logger.info(f'trained: {out}')
        model.save_pretrained(self.peft_output_dir, save_embedding_layers=True)
        logger.info(f'saved peft adapter: {self.peft_output_dir}')
        if self.merged_output_dir is not None:
            model = model.merge_and_unload()
            model.save_pretrained(
                self.merged_output_dir,
                save_embedding_layers=True)
            logger.info(f'saved merged model: {self.merged_output_dir}')
        logger.debug('hf trainer complete')
        return out
