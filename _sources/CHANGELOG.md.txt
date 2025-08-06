# Change Log
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).


## [Unreleased]


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
[Unreleased]: https://github.com/plandes/lmtask/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/plandes/lmtask/compare/v0.0.1...v0.1.0
[0.0.1]: https://github.com/plandes/lmtask/compare/v0.0.0...v0.0.1

[pixi]: https://pixi.sh
[zensols.deeplearn]: https://github.com/plandes/deeplearn
