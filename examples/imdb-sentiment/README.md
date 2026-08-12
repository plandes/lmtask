# IMDB sentiment specialization

This is the canonical end-to-end LMTask example. It specializes the configured
Qwen 3 instruction model for binary IMDB sentiment classification using
PEFT/LoRA supervised fine-tuning.

The configuration is split between:

* [`trainconf/imdb-qwen3.yml`](../../trainconf/imdb-qwen3.yml), which selects
  the Qwen model resources; and
* [`trainconf/imdb-shared.yml`](../../trainconf/imdb-shared.yml), which defines
  the dataset transformation, task templates, and training parameters.


## 1. Install LMTask

From PyPI:

```bash
pip install zensols.lmtask
```

For development, initialize the repository environment using the project's
existing build workflow.


## 2. Review the configuration

The shared configuration downloads `stanfordnlp/imdb`, maps the numeric labels
to `positive` and `negative`, renames `text` to `instruction`, shuffles the
records, and selects a 1,000-record training subset.

The task has separate templates for training and inference. This keeps the
output contract explicit while allowing the training record to include the
expected answer.


## 3. Inspect formatted records

Always inspect at least one record before starting a training run:

```bash
lmtask -c trainconf/imdb-qwen3.yml sample -m 1
```

This verifies dataset access, preprocessing, task formatting, and the selected
model's chat-template behavior.


## 4. Inspect the trainer

```bash
lmtask -c trainconf/imdb-qwen3.yml trainer
```

Use the CLI help if your installed release exposes a different trainer-inspect
subcommand name:

```bash
lmtask --help
```


## 5. Train the PEFT adapter

```bash
lmtask -c trainconf/imdb-qwen3.yml train
```

LMTask constructs a PEFT model and trains its LoRA adapter parameters with TRL
`SFTTrainer`. The original source-model parameters remain frozen during this
optimization.

The trainer writes the adapter to `peft_output_dir`. When
`merged_output_dir` is configured, it also creates a deployment artifact by
merging the trained adapter into a loaded copy of the source model.


## 6. Use the specialized task

The trained model is selected through configuration, so application code keeps
the same request/response interface:

```python
from zensols.lmtask import ApplicationFactory, InstructTaskRequest

factory = ApplicationFactory.get_task_factory()
task = factory.create('sentiment')
response = task.process(InstructTaskRequest(
    instruction='A clever film with a disappointing ending.'))
print(response.model_output_json)
```

The exact model-resource override needed to point at the resulting adapter or
merged model depends on the output paths in your effective configuration. Use
`lmtask -c trainconf/imdb-qwen3.yml trainer` and the generated model result to
confirm those paths rather than copying an assumed local directory.


## 7. Record a reproducible result

Do not publish an accuracy number without its full experiment context. Use
[`BENCHMARKS.md`](../../BENCHMARKS.md) to record:

* source checkpoint and revision;
* dataset revision and split;
* LoRA and SFT parameters;
* software versions;
* GPU and peak memory;
* wall-clock training time;
* adapter and merged-model sizes; and
* held-out accuracy, macro-F1, or other task metrics.


## Other included model configurations

The same IMDB task is available for:

* `trainconf/imdb-llama3.yml`
* `trainconf/imdb-gemma4.yml`
* `trainconf/imdb-dsr1qwen3.yml`

These are model-family configurations, not claims that all checkpoints have the
same hardware requirements or produce equivalent task quality.
