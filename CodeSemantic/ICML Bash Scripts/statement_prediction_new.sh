#!/bin/bash

MODELS=({7..17})
SHOT=0
DATA_ID=0
QUANTIZED_PREDICTION="no"
QUANTIZED_TYPE="None"

INCONTEXT="different"
COT="no"
QUANTIZED_RANDOM="no"
API_DEF="no"

CACHE_PATH="/home/monoshi/.cache/huggingface/hub/*"

echo "Clearing HuggingFace cache..."
rm -rf $CACHE_PATH

cd ..

for model_id in "${MODELS[@]}"; do
    echo "=== Running model_id $model_id ==="
    echo "Running: model=$model_id, shot=$SHOT, data=$DATA_ID, quant=$QUANTIZED_PREDICTION"
    
    python statement_semantic.py \
        --data_id $DATA_ID \
        --model_id $model_id \
        --pt_id 1 \
        --language python \
        --prediction statement \
        --shot $SHOT \
        --incontext $INCONTEXT \
        --CoT $COT \
        --quantized_prediction $QUANTIZED_PREDICTION \
        --quantized_random $QUANTIZED_RANDOM \
        --quantized_type $QUANTIZED_TYPE \
        --API_def $API_DEF

    rm -rf $CACHE_PATH
done

echo "=== All experiments completed! ==="
rm -rf $CACHE_PATH