#!/bin/bash
for condition in natural flattened inverted; do
  for seed in 0 1 2; do
    echo "=== training bengali $condition seed$seed ==="
    python src/models/instrument/train.py \
      --manifest data/manifests/bengali_${condition}.jsonl \
      --script bengali \
      --output-root /content/drive/MyDrive/vlm-ocr-eval/checkpoints \
      --condition $condition --seed $seed --batch-size 32 --total-steps 5000
  done
done
