#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   export DEST=/path/to/dir && bash examples/LIBERO-Mem/data_preparation.sh
# or
#   bash examples/LIBERO-Mem/data_preparation.sh /path/to/dir

DEST="${DEST:-${1:-}}"
if [[ -z "${DEST}" ]]; then
  echo "ERROR: DEST is not set."
  echo "  export DEST=/path/to/dir && bash examples/LIBERO-Mem/data_preparation.sh"
  echo "  or: bash examples/LIBERO-Mem/data_preparation.sh /path/to/dir"
  exit 1
fi

CUR="$(pwd)"
mkdir -p "$DEST"

python -m pip install -U "huggingface-hub==0.35.3"

repo=CollisionCode/libero_mem_lerobot_v2.1
hf download "$repo" --repo-type dataset --local-dir "$DEST/libero-mem/datasets/${repo##*/}" --max-workers 8

init_files=CollisionCode/libero_mem_init_files
hf download "$init_files" --repo-type dataset --local-dir "$DEST/libero-mem/libero/init_files/${init_files##*/}" --max-workers 8

mkdir -p "$CUR/playground/Datasets"
ln -s "$DEST/libero-mem/datasets" "$CUR/playground/Datasets/LEROBOT_LIBERO_MEM_DATA"
LIBERO_MEM_PATH="path/to/libero-mem"
ln -s $DEST/libero-mem/libero/init_files" "$LIBERO_MEM_PATH/libero/libero/init_files/libero_mem"

## move modality
cp "$CUR/examples/LIBERO-Mem/train_files/modality.json" "$CUR/playground/Datasets/LEROBOT_LIBERO_MEM_DATA/libero_10_no_noops_1.0.0_lerobot/meta"
