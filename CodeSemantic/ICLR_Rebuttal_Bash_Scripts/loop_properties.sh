#!/bin/bash

CACHE_PATH="/home/monoshi/.cache/huggingface/hub/*"

cd ..
declare -A DATA_SETTINGS=(
    [13]="iteration"
    [14]="body" 
    [15]="after"
)

MODELS=(23 24)

for model_id in "${MODELS[@]}"; do
    echo "Clearing Hugging Face cache before running model_id $model_id..."
    rm -rf $CACHE_PATH

    for data_id in 13 14 15; do
        settings=${DATA_SETTINGS[$data_id]}
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
done

echo "Cleaning up cache..."
rm -rf $CACHE_PATH
echo "All experiments completed!"