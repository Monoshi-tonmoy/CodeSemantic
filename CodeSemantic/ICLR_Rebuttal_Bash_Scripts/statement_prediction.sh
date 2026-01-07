#!/bin/bash

CACHE_PATH="/home/monoshi/.cache/huggingface/hub/*"

cd ..

for model_id in {8..17}; do
    echo "=== Running model_id $model_id ==="
    rm -rf $CACHE_PATH

    for run in 1 2 3; do
        echo "--- Run $run for model_id $model_id ---"

        for shot in 0; do
            for incontext in "different"; do
                for CoT in "no"; do
                    for quantized in "no"; do
                        python statement_semantic.py \
                            --data_id 0 \
                            --model_id $model_id \
                            --pt_id 1 \
                            --language python \
                            --prediction statement \
                            --shot $shot \
                            --incontext $incontext \
                            --CoT $CoT \
                            --quantized_prediction $quantized \
                            --quantized_random no \
                            --API_def no
                    done
                done
            done
        done
    done
done

model_id=7
echo "=== Running model_id 7 ==="
rm -rf $CACHE_PATH

for run in 1 2 3; do
    echo "--- Run $run for model_id 7 ---"


    for shot in 0; do
        for incontext in "different"; do
            for CoT in "no"; do
                for quantized in "no"; do
                    python statement_semantic.py \
                        --data_id 0 \
                        --model_id $model_id \
                        --pt_id 1 \
                        --language python \
                        --prediction statement \
                        --shot $shot \
                        --incontext $incontext \
                        --CoT $CoT \
                        --quantized_prediction $quantized \
                        --quantized_random no \
                        --API_def no
                done
            done
        done
    done
done

rm -rf $CACHE_PATH
