# Inference

Inference uses the same named task contract that was used to prepare and train
the specialized model. LMTask keeps request formatting, model loading,
generation, response handling, and optional caching behind that task
abstraction.

For introductory command-line and Python examples, see the [README].


## Inference contract

Instruction tasks use a dedicated inference template rather than the training
template:

```yaml
lmtask_task_dataset:
  inference_template: |-
    Classify the financial sentiment of the sentence.
    Only output one label as 'negative', 'neutral', or 'positive'.
    ### Financial sentence:{{ request.instruction }}
    ### Sentiment:
```

The inference template expresses the same task and output contract as training,
but renders request fields rather than expected output fields.


## Request and response handling

Applications invoke a configured task through its request/response API. The
task is responsible for applying the configured prompt contract and returning
the task response in the configured form.

Depending on task configuration, LMTask can return raw text or machine-readable
structured output. Structured-response handling can recover usable data from
partial JSON when generation reaches a token limit.


## Held-out inference

Held-out testing intentionally uses the task's normal inference path. The shared
test configuration uses:

```yaml
lmtask_dataset_test_source:
  class_name: zensols.lmtask.dataset.LoadedTaskDatasetFactory
  task: 'instance: lmtask_task_dataset'
  messages_field: null
```

Leaving `messages_field` unset avoids pre-applying the tokenizer chat template
to the test dataset. Formatting is instead applied for each inference request,
matching application behavior more closely.

Test output preserves source/gold fields and adds the configured prediction
field. Raw model output can also be retained when configured for error
analysis.

This distinction is important:

- validation evaluates the training objective during specialization;
- held-out testing runs the specialized task through its inference path; and
- benchmarking scores those held-out predictions with task-specific metrics.


## Application and agent use

LMTask exposes a trained capability as a normal Python task. An application or
agent framework can call that task without taking ownership of model-specific
prompt construction, adapter loading, response cleanup, or task semantics.

LMTask therefore supplies the specialized capability; an orchestration layer
decides when and how to invoke it.

See [Configuration] for the task contract and [Training] for how the adapter is
specialized.

<!-- links -->

[README]: project:../index.md
[Configuration]: project:configuration.md
[Training]: project:training.md
