#!/bin/bash

CACHE_PATH="/home/monoshi/.cache/huggingface/hub/*"

rm -rf $CACHE_PATH

for model_id in 17; do

    python statement_semantic.py --data_id 5 --model_id $model_id --pt_id 0 --language python --prediction output
    python statement_semantic.py --data_id 5 --model_id $model_id --pt_id 0 --language python --prediction input
    python statement_semantic.py --data_id 9 --model_id $model_id --pt_id 0 --language c --prediction alias
    
    python statement_semantic.py --data_id 6 --model_id $model_id --pt_id 0 --language python --prediction loop --settings iteration
    python statement_semantic.py --data_id 7 --model_id $model_id --pt_id 0 --language python --prediction loop --settings body
    
    rm -rf $CACHE_PATH
    
    echo "Completed all experiments for model $model_id"
done