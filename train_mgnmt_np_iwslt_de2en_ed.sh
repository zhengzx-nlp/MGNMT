#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0,1,2,3

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

MODEL_NAME="IWSLT-DE2EN-MGNMT-NP-TEST"
CONFIG_PATH="./configs/ed/iwslt_de2en_mgnmt_np.yaml"
LOG_PATH=.log/${MODEL_NAME}
# LOG_PATH="./log"
SAVETO=./save/


mkdir -p $SAVETO
echo $MODEL_NAME > $SAVETO/MODEL_NAME

cp ${CONFIG_PATH} ${SAVETO}/${MODEL_NAME}.config

python -m src.bin.train \
               --task "mgnmt" \
               --model_name ${MODEL_NAME} \
               --reload \
               --config_path ${CONFIG_PATH} \
               --log_path ${LOG_PATH} \
               --saveto ${SAVETO} \
               --valid_path "./valid" \
               --use_gpu --debug

