#!/usr/bin/env bash
set -e
if [ -z "$1" ]; then
  echo "Usage: $0 <image-or-dir> [weights]"
  exit 1
fi
IMAGE_OR_DIR=$1
WEIGHTS=${2:-weights/your_weights.pth}
python src/inference.py --image "$IMAGE_OR_DIR" --weights "$WEIGHTS" --out-dir artifacts/
