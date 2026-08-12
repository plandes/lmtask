# Configuration

LMTask configuration connects the model, task contract, datasets, training
behavior, testing, and inference. Project configurations typically import
reusable model and workflow resources and override only task-specific settings.


## Composition

A task configuration can import shared model, training, and testing resources:

```yaml
project:
  import:
    model_imp:
      config_file: 'resource(zensols.lmtask): resources/models/gemma4.conf'
      type: importini
    train_imp:
      config_file: 'resource(zensols.lmtask): resources/train-dataset.yml'
      type: importyaml
    test_imp:
      config_file: 'resource(zensols.lmtask): resources/test-dataset.yml'
      type: importyaml
```

The model resource supplies model- and tokenizer-specific configuration. The
training and testing resources supply shared LMTask components. The project
configuration then specializes those resources for one dataset/model task.

For generated artifacts, identify the dataset and model explicitly:

```yaml
lmtask_dataset:
  dataset_name: fpb
  model_name: gemma4
  is_base: false
```

These values are used to organize runtime data under
`data/<dataset>/<model>/`.


## Dataset contract

Training examples are normalized to fields consumed by the configured task. For
instruction-style tasks, the input is typically stored in `instruction` and
the expected training response in `output`.

For example, an IMDB source maps the numeric sentiment label to task text and
normalizes the source text field:

```yaml
lmtask_dataset_train_source:
  source: stanfordnlp/imdb
  load_args:
    split: train
  pre_process: |-
    ds = ds.map(lambda x: {
        'output': 'positive' if x['label'] == 1 else 'negative'})
    ds = ds.rename_column('text', 'instruction')
```

Preprocessing is also where task configurations can shuffle, subset, rename,
filter, or partition source data. Training, validation, and test partitions
should remain distinct when all three are used.


## Training and inference templates

Instruction configurations define separate templates for supervised training
examples and inference requests:

```yaml
lmtask_task_dataset:
  role: 'You are a financial-news sentiment classifier.'
  train_template: |-
    Classify the financial sentiment of the sentence.
    ### Financial sentence:{{ instruction }}
    ### Sentiment:```{{ output }}```
  inference_template: |-
    Classify the financial sentiment of the sentence.
    ### Financial sentence:{{ request.instruction }}
    ### Sentiment:
```

Both templates encode the same task and output contract. The training template
renders dataset fields, including the expected response. The inference template
renders request fields without the gold response.


## Held-out test source

The shared test configuration leaves chat formatting to the normal inference
path:

```yaml
lmtask_dataset_test_source:
  class_name: zensols.lmtask.dataset.LoadedTaskDatasetFactory
  task: 'instance: lmtask_task_dataset'
  messages_field: null
```

This keeps held-out testing aligned with application inference rather than
pre-formatting the test dataset differently.


## Task-specific overrides

Training and PEFT settings can be overridden independently from the dataset and
prompt contract:

```yaml
lmtask_trainer_hf_training_arguments:
  num_train_epochs: 1
  optim: 'paged_adamw_32bit'
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8

lmtask_trainer_hf_peft:
  r: 128
  lora_alpha: 32
  lora_dropout: 0.05
```

Reusable model-family defaults belong in shared resources; dataset-, task-, and
experiment-specific settings belong in project configuration.

See [Training] for the specialization lifecycle and [Inference] for how the
same task contract is used after training.

<!-- links -->

[Training]: project:training.md
[Inference]: project:inference.md
