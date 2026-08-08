#!/usr/bin/env bash
set -euo pipefail

# A small, recordable walkthrough. Model downloads and GPU requirements depend
# on the effective LMTask configuration.
lmtask task
lmtask instruct sentiment \
  'The interface is excellent, but installation was frustrating.'
lmtask -c trainconf/imdb-qwen3.yml sample -m 1
