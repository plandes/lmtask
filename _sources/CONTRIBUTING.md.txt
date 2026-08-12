# Contributing

If you discover issues, have ideas for improvements or new features, please
report them to the [issue tracker][1] of the repository or submit a pull
request. Please, try to follow these guidelines when you do so.


## Report an issue

Before opening an issue:

* check whether it has already been reported;
* reproduce it with the latest release or `master` when practical; and
* reduce the problem to the smallest task and configuration that still fails.

For model-specific problems, include:

* LMTask version or Git commit;
* Python, PyTorch, Transformers, TRL, PEFT, and Accelerate versions;
* operating system, GPU, CUDA version, and peak memory when relevant;
* exact model checkpoint and revision;
* quantization configuration;
* the smallest relevant task configuration; and
* the full traceback or incorrect response.

Do not include access tokens, private model credentials, protected data, or
licensed dataset content in an issue.


## Add a model integration

A model-family contribution should include:

1. an inference resource configuration;
2. a PEFT training configuration when the model supports training;
3. tokenizer and chat-template settings;
4. any model-specific generator, trainer, or output-cleanup implementation;
5. focused tests that do not require publishing credentials; and
6. documentation of the checkpoint, upstream license, hardware, and known
   limitations.

Prefer extending the existing model-resource interfaces over adding
model-specific conditionals to shared task code.


## Add a task example

A useful task example contains:

* a narrow task and explicit output contract;
* separate training and inference templates when applicable;
* deterministic preprocessing with a recorded seed;
* a small sample configuration suitable for inspection;
* expected structured output; and
* a short explanation of how the task can be called from application code.

Published benchmark numbers must follow [`BENCHMARKS.md`](BENCHMARKS.md).


## Pull requests

* Use a topic branch.
* Keep each pull request focused on one change.
* Follow the existing code and configuration conventions.
* Add or update tests for behavioral changes.
* Run `make test` before submitting.
* Update the README, documentation, and changelog when user-visible behavior
  changes.
* Use clear commit messages and reference related issues.
* Squash fixup commits when appropriate.


## Development workflow

The project uses its existing Relpo/Pixi build workflow. Common targets include:

```bash
make pyinit
make info
make test
```

See the repository `makefile` and generated project metadata for additional
build, documentation, and release targets.


## Code of conduct

Be precise, constructive, and respectful. Technical disagreement is welcome;
personal attacks and harassment are not.


## Guidelines

* Read [how to properly contribute to open source projects on Github][2].
* Use a topic branch to easily amend a pull request later, if necessary.
* Use the same coding conventions as the rest of the project.
* Make sure that the unit tests are passing (`make test`).
* Write [good commit messages][3].
* Mention related tickets in the commit messages (e.g. `[Fix #N] Add command ...`).
* Update the [changelog][6].
* [Squash related commits together][5].
* Open a [pull request][4] that relates to *only* one subject with a clear title
  and description in grammatically correct, complete sentences.


[1]: https://github.com/plandes/lmtask/issues
[2]: http://gun.io/blog/how-to-github-fork-branch-and-pull-request
[3]: http://tbaggery.com/2008/04/19/a-note-about-git-commit-messages.html
[4]: https://help.github.com/articles/using-pull-requests
[5]: http://gitready.com/advanced/2009/02/10/squashing-commits-with-rebase.html
[6]: https://github.com/plandes/lmtask/blob/master/CHANGELOG.md
