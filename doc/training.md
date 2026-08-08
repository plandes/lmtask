# Training

LMTask specializes a configured task with supervised post-training using
PEFT/LoRA and TRL `SFTTrainer`. During LoRA optimization, the source model
weights remain frozen and the learned adapter parameters carry the task
specialization.


## Training lifecycle

The configured workflow is:

1. load and preprocess the training dataset;
2. render examples with the task training template;
3. apply model/tokenizer formatting;
4. optimize the PEFT/LoRA adapter;
5. evaluate trainer diagnostics on the validation split; and
6. persist the adapter and LMTask training result.

The training and inference templates are part of the same named task contract,
so specialization does not end at the adapter artifact.


## Training, validation, and test data

The three dataset roles are intentionally distinct:

| Split      | Purpose                                                   |
|------------|-----------------------------------------------------------|
| Training   | Optimize the PEFT/LoRA adapter                            |
| Validation | Report trainer diagnostics during specialization          |
| Test       | Measure final task behavior through normal task inference |

For the Financial PhraseBank benchmark, the configured split is deterministic
and stratified: 1,752 training examples, 256 validation examples, and 256
held-out test examples.

Trainer statistics such as validation loss and token accuracy describe the SFT
objective. Task-level metrics are computed separately from held-out inference.


## PEFT and trainer settings

Task configurations can override shared PEFT defaults:

```yaml
lmtask_trainer_hf_peft:
  r: 128
  lora_alpha: 32
  lora_dropout: 0.05
```

Trainer arguments are configured separately:

```yaml
lmtask_trainer_hf_training_arguments:
  num_train_epochs: 1
  optim: 'paged_adamw_32bit'
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
```

Model-family resources provide the compatible model, tokenizer, target modules,
quantization behavior, and other shared defaults. Project configuration records
intentional task-specific changes.


## Persisted training state

The normal specialization artifact is a PEFT adapter. LMTask also persists a
training result containing the framework-level training state and statistics.

Underlying trainer checkpoints and the final adapter are separate concerns:
trainer checkpoints reflect checkpoint policy during optimization, while the
final PEFT artifact is the reusable specialization consumed by later task
inference.

Held-out test results and benchmark reports are produced after specialization
and are not training-time validation artifacts.

See [Configuration] for task and dataset setup and [Inference] for the
post-training task contract.

<!-- links -->

[Configuration]: project:configuration.md
[Inference]: project:inference.md
