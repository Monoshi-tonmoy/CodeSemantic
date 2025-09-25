#!/bin/bash

cd ..

# for model_id in 20; do
#     for shot in 3; do

#         python statement_semantic.py \
#             --data_id 0 \
#             --model_id $model_id \
#             --pt_id 1 \
#             --language python \
#             --prediction statement \
#             --shot $shot \
#             --incontext different \
#             --CoT no \
#             --quantized_prediction no \
#             --quantized_random no \
#             --API_def no
#     done
# done

for model_id in 3; do
    for shot in 0 1 3; do
        if [ "$shot" -eq 0 ]; then
            for quantized_random in yes no; do
                python statement_semantic.py \
                    --data_id 10 \
                    --model_id $model_id \
                    --pt_id 1 \
                    --language python \
                    --prediction statement \
                    --shot $shot \
                    --incontext different \
                    --CoT no \
                    --quantized_prediction yes \
                    --quantized_random $quantized_random \
                    --API_def no
            done
        else
            python statement_semantic.py \
                --data_id 10 \
                --model_id $model_id \
                --pt_id 1 \
                --language python \
                --prediction statement \
                --shot $shot \
                --incontext different \
                --CoT no \
                --quantized_prediction yes \
                --quantized_random no \
                --API_def no
        fi
    done
done