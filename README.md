# Inferencing and Training Large Language Model Tasks

[![PyPI][pypi-badge]][pypi-link]
[![Python 3.11][python311-badge]][python311-link]

A large language model (LLM) API to train and inference specifically for tasks.
The API provides utility classes and configuration to streamline project's
access to LLM responses that (can be) machine readable, such as querying the
LLM to produce JSON output--even if that output is partial given output token
limits.  The package provides an API to train and interface with LLMs as both
pretrained embeddings and instruct models.

Features:

* Create new LLMs with configuration without having to write code.
* [Three examples](resources/tasks.yml) of "code-less" models: sentiment
  analysis, NER tagging and generation.
* Cache LLM responses to save from recomputing potentially costly prompts
  (optional feature).
* [Command-line](#command-line) interface to inference, pre-train and
  post-train LLM models.
* [Advanced API](#python-api) to read responses and accept partial output for
  max token cutoffs.
* Chat template integration when supported.
* Extendable interfaces with LLMs with built in support for Llama 3.
* [Easy to configure datasets](#datasets) processed by model trainers


## Documentation

See the [full documentation](https://plandes.github.io/lmtask/index.html).
The [API reference](https://plandes.github.io/lmtask/api.html) is also
available.


## Obtaining

The library can be installed with pip from the [pypi] repository:
```bash
pip3 install zensols.lmtask
```

A Conda environment can also be created with the
[environment.yml](src/python/environment.yml):
```bash
conda env create -f src/python/environment.yml
```


## Usage

The package can be used from the command line to both inference and train a new
model or as an API.


### Command Line

The command-line can be used for inferencing to list available tasks:

```bash
lmtask task
```

Generate text by inferencing using the Llama base model:

```bash
lmtask stream base_generate 'in a world long long away' \
  --override=lmtask_model_generate_args.temperature=0.9
```

Use named entity recognition (NER):
```bash
lmtask instruct ner 'UIC is in Chicago.' 
model_output_json:
    label: ORG
    span: [0, 3]
    text: UIC
    label: O
    span: [4, 6]
    text: is
    label: O
    span: [7, 9]
    text: in
    label: LOC
    span: [10, 15]
    text: Chicago
```

The command-line program can be used to train new models with just
configuration.  See the [trainconf](trainconf) directory for examples of
configuration file.  Before you train, you might want to get a sample of the
configured dataset:

```bash
lmtask -c trainconf/imdb.yml sample -m 1
________________________________________
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 26 Jul 2024

You are a sentiment classifier.<|eot_id|><|start_header_id|>user<|end_header_id|>

Only output the sentiment.

### Review:We always watch American movies with their particular...
### Sentiment:```positive```<|eot_id|>
```

To train a new sentiment model on the IMDB dataset:

```bash
lmtask -c trainconf/imdb.yml train
```


### Python API

The Python API can be used to access tasks directly.

```python
>>> import json
>>> from zensols.lmtask import ApplicationFactory
>>> from zensols.lmtask import InstructTaskRequest

# create the task factory
>>> fac = ApplicationFactory.get_task_factory()

# list configured tasks
>>> fac.write(short=True)
base_generate (base generate text)
instruct_generate (base generate text)
ner (tags named entities)
sentiment (classifies sentiment)

# create a sentiment analysis task
>>> task = fac.create('sentiment')

# inference
>>> sents = 'I love football.\nI hate olives.\nEarth is big.'
>>> res = task.process(InstructTaskRequest(instruction=sents))

# print the JSON result as formatted text
>>> print(json.dumps(res.model_output_json, indent=4))
[
    {
        "index": 0,
        "sentence": "I love football.",
        "label": "+"
    },
    {
        "index": 1,
        "sentence": "I hate olives.",
        "label": "-"
    },
    {
        "index": 2,
        "sentence": "Earth is big.",
        "label": "n"
    }
]
```


## Datasets

The package features easy to configure datasets and data processing on it.  For
example, the following is taken from the [IMDB training
configuration](trainconf/imdb.yml) example:

```yaml
# a dataset factory instance used by the trainer (lmtask_trainer_hf)
lmtask_imdb_source:
  class_name: zensols.lmtask.dataset.LoadedTaskDatasetFactory
  # the dataset name (downloaded if not already); this can be a `pathlib.Path`,
  # Pandas dataframe or Zensols Stash
  source: stanfordnlp/imdb
  # use only the training split
  load_args:
    split: train
  # the task that consumes the data, which will format each datapoint
  # specifically for that task's model
  task: 'instance: lmtask_task_imdb'
  # preprocessing Python source code to add labels and subset the data (db.select)
  pre_process: |-
    ds = ds.map(lambda x: {'output': 'positive' if x['label'] == 1 else 'negative'})
    ds = ds.rename_column('text', 'instruction')
    ds = ds.shuffle(seed=0)
    # 7K takes 55m
    ds = ds.select(range(7_000))
```


## Changelog

An extensive changelog is available [here](CHANGELOG.md).


## Community

Please star this repository and let me know how and where you use this API.
Contributions as pull requests, feedback and any input is welcome.


## License

[MIT License](LICENSE.md)

Copyright (c) 2025 Paul Landes


<!-- links -->
[pypi]: https://pypi.org/project/zensols.lmtask/
[pypi-link]: https://pypi.python.org/pypi/zensols.lmtask
[pypi-badge]: https://img.shields.io/pypi/v/zensols.lmtask.svg
[python311-badge]: https://img.shields.io/badge/python-3.11-blue.svg
[python311-link]: https://www.python.org/downloads/release/python-3110
