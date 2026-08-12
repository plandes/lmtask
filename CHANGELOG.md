# Change Log
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).


## [Unreleased]


## [0.2.0] - 2026-08-12
### Removed
- Support for Python 3.11.
- Merged embeddings output.  Base and PEFT models are still written to disk.

### Added
- New models: Qwen3, Gemma4, DeepSeek R1 Distill Qwen
- Support for Python 3.12 and 3.13
- Basic model testing (evaluation) functionality.
- Model benchmarking library and CLI.
- Canonical IMDB sentiment specialization tutorial.
- Framework-neutral agent-tool integration example.
- Benchmark and reproducibility protocol.
- GitHub presentation guidance, social-preview source, and issue templates.
- Report output with [zensols.datdesc] for LaTeX, Excel, and table JSON report
  output.

### Changed
- Reposition the README around LMTask's complete dataset-to-PEFT-to-task
  lifecycle.
- Correct training terminology to parameter-efficient supervised fine-tuning.
- Expand package metadata and search keywords.
- Improve contribution guidance for model integrations and task examples.
- Replace the Llama specific generator with a generic code + config based
  class.
- Fix Sphinx build by adding `numpy` dependency.
- Move Llama configuration to `resources/models`.
- Copy `zensols.deeplearn.torchconfig` (rather than add heavy dependency).
  This was added for reproducibility of experiments.
- Upgrade dependencies:
  - `transformers`: 5.5
  - `torch`: 2.10
  - `bitsandbytes`: 0.49
  - `xformers`: 0.0.35


## [0.1.1] - 2025-08-20
### Changed
- Pickled training results go to the Peft output model's directory.
- Training steps configuration was commented out to allow client projects to
  keep the HF library defaults.


## [0.1.0] - 2025-08-06
### Removed
- Unsloth configuration and API.

### Changed
- Switch build tools to [pixi].
- Removed [zensols.deeplearn] dependency.
- Upgrade dependencies: `torch` `transformers`, `datasets`, `accelerate`,
  `peft`, `trl`.
- Removed default quantization configuration.
- Add LoRA generator configuration.
- Separate and configure LoRA adapter output directory.
- Add trainer evaluation dataset.
- Disable adding end of sentence token in `GenerateTask` (turn on by setting
  `train_add_eos` is `True`).  Now the `SFTTrainer` does this by default.
- `InstructTask.apply_chat_template` defaults to ``False``.  Now the
  `SFTTrainer` does this by default.


## [0.0.1] - 2025-05-04
### Added
- Initial version.


<!-- links -->
[Unreleased]: https://github.com/plandes/lmtask/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/plandes/lmtask/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/plandes/lmtask/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/plandes/lmtask/compare/v0.0.1...v0.1.0
[0.0.1]: https://github.com/plandes/lmtask/compare/v0.0.0...v0.0.1

[pixi]: https://pixi.sh
[zensols.deeplearn]: https://github.com/plandes/deeplearn
[zensols.datdesc]: https://github.com/plandes/datdesc
