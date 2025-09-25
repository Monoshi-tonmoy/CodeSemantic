#!/bin/bash

CACHE_PATH="/home/monoshi/.cache/huggingface/hub/*"
cd ..

for model_id in {9..17}; do
    echo "Clearing Hugging Face cache before running model_id $model_id with shot $shot..."
    rm -rf $CACHE_PATH
    for shot in 0 1 2 3; do
        python statement_semantic.py \
            --data_id 1 \
            --model_id $model_id \
            --pt_id 1 \
            --language c \
            --prediction statement \
            --shot $shot \
            --incontext different \
            --CoT no \
            --quantized_prediction no \
            --quantized_random no \
            --API_def no
    done
done

rm -rf $CACHE_PATH
echo "Cleared Hugging Face cache after all runs."