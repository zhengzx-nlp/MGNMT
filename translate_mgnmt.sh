#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0


MODEL_NAME="`cat ./save/MODEL_NAME`"
MODEL_PATH=./save/$MODEL_NAME.best.final
CONFIG_PATH=./save/$MODEL_NAME.config
SOURCE_PATH="/home/zzheng/data/mt/WMT14-EN-DE/newstest2014.de.norm.tok"
TARGET_PATH="/home/zzheng/data/mt/WMT14-EN-DE/newstest2014.en.norm.tok"
SRC_LANG="de"
TGT_LANG="en"

SRC_LANG_="src"

REFERENCE_PATH="/home/zzheng/data/mt/WMT14-EN-DE/newstest2014.en"
# REFERENCE_PATH="/home/zzheng/data/mt/WMT14-EN-DE/newstest2014.de"

SAVETO=./results/
mkdir -p $SAVETO
saveto=$SAVETO/test.trans

python -m src.bin.translate_mgnmt \
    --model_name $MODEL_NAME \
    --task "mgnmt" \
    --source_path $SOURCE_PATH \
    --model_path $MODEL_PATH \
    --config_path $CONFIG_PATH \
    --saveto $saveto \
    --batch_size 2048 \
    --beam_size 4 \
    --alpha 0.6 \
    --beta 0.0 \
    --reranking --gamma 0.0 \
    --src_lang ${SRC_LANG_} \
    --keep_n 1 \
    --use_gpu

perl ./scripts/postprocess.sh $TGT_LANG < $saveto.0 > $saveto.0.post
BLEU=`sacrebleu -tok intl -lc $REFERENCE_PATH < $saveto.0.post`
echo $BLEU
echo "$saveto.0.post:    $BLEU" >> ./results/bleu.log
