#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/synthetic_inputs}"
RESOLUTION="${RESOLUTION:-64}"
COUNTS=(1000 2000 4000 8000 12000 16000 20000 26000)

cd "${REPO_ROOT}"

for COUNT in "${COUNTS[@]}"; do
  "${PYTHON_BIN}" script_modeling/create_synthetic_inference_dataset.py \
    --output-dir "${OUTPUT_ROOT}/R${COUNT}" \
    --raw-images "${COUNT}" \
    --resolution "${RESOLUTION}" \
    --remake
done
