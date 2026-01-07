#!/bin/bash

CACHE_PATH="/home/monoshi/.cache/huggingface/hub/*"

cd ..

for model_id in {8..9}; do
    echo "Clearing Hugging Face cache before running model_id $model_id..."
    rm -rf $CACHE_PATH

    for prediction in "output" "input"; do
        python statement_semantic.py \
            --data_id 30 \
            --model_id $model_id \
            --pt_id 1 \
            --language python \
            --prediction $prediction \
            --shot 0 \
            --incontext different \
            --CoT no \
            --quantized_prediction no \
            --quantized_random no \
            --API_def no
    done
done

# Final cache cleanup
echo "Final cache cleanup..."
rm -rf $CACHE_PATH