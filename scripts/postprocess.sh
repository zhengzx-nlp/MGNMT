#!/usr/bin/env bash

# Usage:
#   bash ./scripts/postprocess.sh <lang-flag>
# Example:
#   bash ./scripts/postprocess.sh zh < your_file.tok > your_file.detok

perl ./src/metric/scripts/recaser/detruecase.perl | \
perl ./src/metric/scripts/tokenizer/detokenizer.perl -l $1