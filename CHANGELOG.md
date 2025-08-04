# Change Log
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).


## [Unreleased]


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


## [0.0.1] - 2025-05-04
### Added
- Initial version.


<!-- links -->
[Unreleased]: https://github.com/plandes/lmtask/compare/v0.0.1...HEAD
[0.0.1]: https://github.com/plandes/lmtask/compare/v0.0.0...v0.0.1

[pixi]: https://pixi.sh
[zensols.deeplearn]: https://github.com/plandes/deeplearn
