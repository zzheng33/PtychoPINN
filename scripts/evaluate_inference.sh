#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  ./evaluate_inference.sh [recon_npz]
  ./evaluate_inference.sh [dataset] [model_key]

Examples:
  ./evaluate_inference.sh inference_outputs/W_PS_W_recon.npz
  ./evaluate_inference.sh W PS_W
  ./evaluate_inference.sh TP1 PS_TP1
  ./evaluate_inference.sh W Unified

Notes:
  - Main metric: ptychopinn_frc_auc_0_to_0.5, higher is better.
  - If no arguments are given, defaults to TP1 PS_TP1.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

TARGET="${1:-TP1}"
MODEL_KEY="${2:-}"

cd "$REPO_ROOT"

if [[ "$TARGET" == *.npz || "$TARGET" == inference_outputs/*.npz ]]; then
  RECON_NPZ="$TARGET"
else
  DATASET="$TARGET"
  if [[ -z "$MODEL_KEY" ]]; then
    MODEL_KEY="PS_${DATASET}"
  fi
  RECON_NPZ="inference_outputs/${DATASET}_${MODEL_KEY}_recon.npz"
fi

if [[ ! -f "$RECON_NPZ" ]]; then
  echo "Missing reconstruction file: $RECON_NPZ" >&2
  echo "Run inference first, for example:" >&2
  echo "  ./run_inference.sh W cuda PS_W" >&2
  exit 1
fi

conda run -n ptychopinn_torch python scripts/evaluate_inference.py "$RECON_NPZ"
