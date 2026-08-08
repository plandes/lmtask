# Benchmarks

LMTask benchmarks evaluate a specialized model through the same task inference
path used by applications. Training-time validation statistics are reported
separately from held-out task metrics.

Detailed benchmark reports are generated per task/model combination. This file
is the stable benchmark index and usage guide; the full generated reports live
under [`benchmarks/`](benchmarks/).

## Results

| Task                 | Model   | Test examples | Accuracy |   Macro-F1 | Report                                         |
|----------------------|---------|--------------:|---------:|-----------:|------------------------------------------------|
| Financial PhraseBank | Gemma 4 |           256 |   0.9531 | **0.9434** | [details](benchmarks/fpb-gemma4/fpb_gemma4.md) |

## Run a benchmark

Benchmark behavior, including task-specific metric calculation, is configured
in `trainconf/*`. For example:

```bash
lmtask -c trainconf/fpb-gemma4.yml benchmark
```

The benchmark action:

1. trains the configured PEFT model when a persisted training result is not
   already available;
2. runs the configured held-out test task when a persisted test result is not
   already available;
3. scores predictions with the metric calculator configured for that task;
4. captures training, testing, environment, Git, CUDA/GPU, and artifact
   metadata; and
5. renders machine-readable and human-readable benchmark artifacts.

The metric layer is task-specific. Classification tasks can report accuracy,
micro/macro/weighted precision, recall and F1, per-class metrics, and invalid
prediction counts. Other task types can select their own scorer without
changing the benchmark runner.

## Runtime artifacts

Training and testing state remains with the model data:

```text
data/<dataset>/<model>/
├── model/
│   ├── base/
│   ├── peft/
│   └── result/
│       ├── train-result.dat
│       └── test-result.dat
└── benchmark/
    ├── <task>_<model>.jsonl
    ├── <task>_<model>.json
    └── <task>_<model>.md
```

`model/base/` contains trainer checkpoints, `model/peft/` contains the final
PEFT adapter, and `model/result/` contains persisted LMTask train/test results.

The `benchmark/` directory is created only by the benchmark workflow:

* `.jsonl` contains row-level gold values and predictions;
* `.json` is the structured benchmark source of truth; and
* `.md` is the rendered human-readable report.

Without running `benchmark`, `test-result.dat` and the `benchmark/` artifacts
are not created.

## Published benchmark artifacts

Large model artifacts and runtime state under `data/` should not be committed
to Git. Publish durable benchmark evidence separately under:

```text
benchmarks/
└── <task>-<model>/
    ├── <task>_<model>.json
    ├── <task>_<model>.md
    └── <task>_<model>.jsonl
```

The Markdown and JSON files should normally be committed. The JSONL prediction
file is optional: commit it only when the source data is redistributable and
the file is small enough to be useful in the repository.

## Reproducibility

A benchmark record should capture, where applicable:

* LMTask Git revision and working-tree state;
* task/model configuration;
* train, validation, and test split sizes and class distributions;
* training time, training loss, global steps, and trainer diagnostics;
* held-out test time and number of evaluated examples;
* task-specific metrics and per-class results;
* invalid/unparseable prediction count;
* PEFT adapter size;
* Python and relevant package versions;
* operating system and kernel;
* visible CUDA devices, GPU model and memory;
* NVIDIA driver and PyTorch CUDA runtime; and
* peak allocated/reserved GPU memory.

For PEFT runs, adapter size refers to the saved adapter payload rather than a
merged model or frozen base-model weights.

See the generated report for each task/model under [`benchmarks/`](benchmarks/)
for the full configuration, environment, training diagnostics, and detailed
metrics.
