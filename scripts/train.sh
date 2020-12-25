#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=2
echo "Using GPU $CUDA_VISIBLE_DEVICES..."
python -m src.bin.train \
			   --model_name "transformer" \
			   --reload \
			   --config_path "./configs/transformer_wmt15_en2fr.yaml" \
			   --log_path "./log" \
			   --saveto "./save" \
			   --valid_path "./valid" \
			   --use_gpu
