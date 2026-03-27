#!/bin/bash


MODELS=({7..17})
SHOTS=({0..3})
INCONTEXTS=("different")  
COTS=("no")              
QUANTIZED_RANDOMS=("no") 
API_DEFS=("no")        

DATA_CONFIGS=(
    "0 no None"
    "10 yes value"
    "31 yes dtype"
)


CACHE_PATH="/home/monoshi/.cache/huggingface/hub/*"
echo "Clearing HuggingFace cache..."
rm -rf $CACHE_PATH

cd ..


for model_id in "${MODELS[@]}"; do
    echo "=== Running model_id $model_id ==="
    
    for shot in "${SHOTS[@]}"; do
        echo "--- Shot $shot ---"
        
        for incontext in "${INCONTEXTS[@]}"; do
            for CoT in "${COTS[@]}"; do
                for quantized_random in "${QUANTIZED_RANDOMS[@]}"; do
                    for API_def in "${API_DEFS[@]}"; do
                        for config in "${DATA_CONFIGS[@]}"; do
                            read data_id quantized_prediction quantized_type <<< "$config"
                            
                            echo "Running: model=$model_id, shot=$shot, data=$data_id, quant=$quantized_prediction"
                            
                            python statement_semantic.py \
                                --data_id $data_id \
                                --model_id $model_id \
                                --pt_id 1 \
                                --language python \
                                --prediction statement \
                                --shot $shot \
                                --incontext $incontext \
                                --CoT $CoT \
                                --quantized_prediction $quantized_prediction \
                                --quantized_random $quantized_random \
                                --quantized_type $quantized_type \
                                --API_def $API_def
                        done
                    done
                done
            done
        done
    done
    rm -rf $CACHE_PATH
done

echo "=== All experiments completed! ==="
rm -rf $CACHE_PATH