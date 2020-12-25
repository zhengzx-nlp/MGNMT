#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

MODEL_NAME="IWSLT-DE2EN-MGNMT"
CONFIG_PATH="./configs/test_mgnmt.yaml"
LOG_PATH=/home/zzheng/experiments/njunmt/mgnmt/.log/${MODEL_NAME}
SAVETO=./save/
#PRETRAIN_MODEL=/home/user_data/zhengzx/models/mt/nist_zh2en_bpe_hybrid_gru_attn_zeroinit_base/save/hybrid-nist_zh2en_bpe-base-gru_attn_zero.best.final

mkdir -p $SAVETO
echo $MODEL_NAME > $SAVETO/MODEL_NAME

cp ${CONFIG_PATH} ${SAVETO}/${MODEL_NAME}.config

python -m src.bin.train \
               --task "mgnmt" \
               --model_name ${MODEL_NAME} \
               --reload \
               --config_path ${CONFIG_PATH} \
               --log_path "./log" \
               --saveto ${SAVETO} \
               --valid_path "./valid" \
               --use_gpu \
               --debug

