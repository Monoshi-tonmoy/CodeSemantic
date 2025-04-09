import numpy as np
from collections import defaultdict
import subprocess
from utils import (
    load_my_dataset,
    load_model,
    load_pt,
    save_jsonl,
    save_results_to_json,
    get_default_config
)
import torch

def evaluate_statement_based(res):
    is_correct = []
    type_correct = defaultdict(int)
    type_total = defaultdict(int)
    language = None

    for d in res:
        try:
            language = d['ori_task']['Programming Language'].lower() 
            stmt_type = d['ori_task']['Statement Type']
            
            if 'pred_ans' in d and len(d['pred_ans']) > 0:
                original_value = d['ori_task']['Value After Statement Execution']
                predicted_value = d['pred_ans'][0] if d['pred_ans'] else None

                if isinstance(original_value, str):
                    original_value = original_value.rstrip(';').strip()
                if isinstance(predicted_value, str):
                    predicted_value = predicted_value.rstrip(';').strip()

                correct = original_value == predicted_value
                is_correct.append(correct)
                type_correct[stmt_type] += int(correct)
                type_total[stmt_type] += 1
            else:
                is_correct.append(False)
                type_total[stmt_type] += 1
        except KeyError as e:
            print(f"Warning: Missing key {e} in response for model {model.model_name}")
            is_correct.append(False)
            if 'stmt_type' in locals():
                type_total[stmt_type] += 1
        except RuntimeError as e:
            if 'CUDA out of memory' in str(e):
                print(f"CUDA out of memory error occurred - treating as incorrect answer")
                is_correct.append(False)
                if 'stmt_type' in locals():
                    type_total[stmt_type] += 1
                if hasattr(torch, 'cuda'):
                    torch.cuda.empty_cache()
            else:
                raise 

    overall_accuracy = np.mean(is_correct) if is_correct else 0.0
    type_accuracy = {t: type_correct[t]/type_total[t] if type_total[t] > 0 else 0.0 
                    for t in type_total}
    
    return overall_accuracy, type_accuracy, dict(type_total), language

def evaluate_block_based(res):
    block_results = defaultdict(lambda: {'correct': [], 'total': 0})
    language = None

    for d in res:
        try:
            language = d['ori_task']['Programming Language'].lower()
            block_size = d['ori_task'].get('Block_Size', 1)
            
            if 'pred_ans' in d and len(d['pred_ans']) > 0:
                original_value = d['ori_task']['Value After Statement Execution']
                predicted_value = d['pred_ans'][0] if d['pred_ans'] else None
                
                correct = original_value == predicted_value
                block_results[block_size]['correct'].append(correct)
                block_results[block_size]['total'] += 1
            else:
                block_results[block_size]['correct'].append(False)
                block_results[block_size]['total'] += 1
        except KeyError as e:
            print(f"Warning: Missing key {e}")
            block_results[block_size]['correct'].append(False)
            block_results[block_size]['total'] += 1
        except RuntimeError as e:
            if 'CUDA out of memory' in str(e):
                print("CUDA out of memory error - treating as incorrect")
                block_results[block_size]['correct'].append(False)
                block_results[block_size]['total'] += 1
                if hasattr(torch, 'cuda'):
                    torch.cuda.empty_cache()
            else:
                raise

    accuracy_results = {
        size: {
            'accuracy': np.mean(results['correct']) if results['correct'] else 0.0,
            'correct': sum(results['correct']),
            'total': results['total'],
        }
        for size, results in block_results.items()
    }
    
    overall_accuracy = np.mean([
        acc for size, results in block_results.items()
        for acc in results['correct']
    ]) if block_results else 0.0
    
    return overall_accuracy, accuracy_results, language

def main(data_id, model_id, pt_id):
    config = get_default_config()
    dataset = load_my_dataset(data_id)
    model = load_model(model_id)
    model.init_ai_kwargs(config)
    pt = load_pt(pt_id)
    
    
    res = model.chat_batch(pt, dataset)
    
    
    is_block_based = 'Block_Size' in dataset[0] if dataset else False
    
    if is_block_based:
        overall_accuracy, block_results, language = evaluate_block_based(res)
        
        save_results_to_json(
            model.model_name, pt_id, language, 
            overall_accuracy, None, block_results,
            is_block_based=True
        )
        
        print(f"Results for {model.model_name} (PT {pt_id}, {language}):")
        print(f"  Overall accuracy: {overall_accuracy:.2f}")
        for block_size, results in sorted(block_results.items()):
            print(f"  Block size {block_size}: {results['accuracy']:.2f} ({results['total']} samples)")
    else:
        overall_accuracy, type_accuracy, type_total, language = evaluate_statement_based(res)
        
        save_results_to_json(
            model.model_name, pt_id, language, 
            overall_accuracy, type_accuracy, type_total
        )
        
        print(f"Results for {model.model_name} (PT {pt_id}, {language}):")
        print(f"  Overall accuracy: {overall_accuracy:.2f}")
        for stmt_type, acc in type_accuracy.items():
            print(f"  {stmt_type}: {acc:.2f} ({type_total[stmt_type]} samples)")

def clear_hf_cache():
    cache_path = "/home/monoshi/.cache/huggingface/hub/*"
    try:
        subprocess.run(f"rm -rf {cache_path}", shell=True, check=True)
        print("✅ Hugging Face cache cleared successfully.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to clear cache: {e}")

if __name__ == '__main__':
    data_id = 4  
    pt_id = 0   
    
    for model_id in range(10, 17): 
        print(f"\n=== Running Model ID: {model_id} ===")
        main(data_id, model_id, pt_id)
        clear_hf_cache()