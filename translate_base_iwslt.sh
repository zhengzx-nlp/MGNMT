#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1


MODEL_NAME="`cat ./save/MODEL_NAME`"
MODEL_PATH=./save/$MODEL_NAME.best.final
CONFIG_PATH=./save/$MODEL_NAME.config
SOURCE_PATH="/home/user_data55/zhengzx/data/mt/IWSLT16-DE-EN/test.de.norm.tok"
REFERENCE_PATH="/home/user_data55/zhengzx/data/mt/IWSLT16-DE-EN/test.en"
TGT_LANG="en"

SAVETO=./results/
mkdir -p $SAVETO
saveto=$SAVETO/test.trans

python -m src.bin.translate \
    --model_name $MODEL_NAME \
    --task "baseline" \
    --source_path $SOURCE_PATH \
    --model_path $MODEL_PATH \
    --config_path $CONFIG_PATH \
    --saveto $saveto \
    --batch_size 150 \
    --beam_size 4 \
    --alpha 0.6 \
    --keep_n 1 \
    --use_gpu

perl ./scripts/postprocess.sh en < $saveto.0 > $saveto.0.post
BLEU=`sacrebleu -lc -tok intl $REFERENCE_PATH < $saveto.0.post`
echo $BLEU
echo "$saveto.0.post:    $BLEU" >> ./results/bleu.log
