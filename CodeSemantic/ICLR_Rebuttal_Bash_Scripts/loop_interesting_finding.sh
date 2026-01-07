#!/bin/bash

CACHE_PATH="/home/monoshi/.cache/huggingface/hub/*"

cd ..

MODELS=(7 8 9 10 11 12 13 14 15 16 17)

for model_id in "${MODELS[@]}"; do
    echo "Clearing Hugging Face cache before running model_id $model_id..."
    rm -rf $CACHE_PATH

    data_id=13
    settings="iteration"
    echo "Running model_id $model_id with data_id $data_id ($settings)..."
    
    python statement_semantic.py \
        --data_id $data_id \
        --model_id $model_id \
        --pt_id 1 \
        --language python \
        --prediction loop \
        --settings $settings \
        --shot 3 \
        --incontext different \
        --CoT no \
        --quantized_prediction no \
        --quantized_random no \
        --API_def no
done

echo "Cleaning up cache..."
rm -rf $CACHE_PATH
echo "All experiments completed!"