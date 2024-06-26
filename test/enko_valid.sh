#!/bin/bash
MODEL_FN=$1
GPU_ID=-1
BEAM_SIZE=1
TEST_FN=./corpus.shuf.valid.tok.bpe.head-1000.en
REF_FN=./corpus.shuf.valid.tok.bpe.head-1000.detok.tok.ko

cat ${TEST_FN} | python ../translate.py --model ${MODEL_FN} --gpu_id ${GPU_ID} --lang enko --beam_size ${BEAM_SIZE} | python ../nlp_preprocessing/detokenizer.py | mecab -O wakati | ./multi-bleu.perl ${REF_FN}
