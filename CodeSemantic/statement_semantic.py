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

def main(data_id, model_id, pt_id):
    config = get_default_config()
    dataset = load_my_dataset(data_id)[:50]
    model = load_model(model_id)
    model.init_ai_kwargs(config)
    pt = load_pt(pt_id)
    
    res = model.chat_batch(pt, dataset)
    
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
    

    save_results_to_json(
        model.model_name, pt_id, language, 
        overall_accuracy, type_accuracy, dict(type_total)
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
    data_id = 0  
    pt_id = 0   
    
    for model_id in range(14, 15): 
        print(f"\n=== Running Model ID: {model_id} ===")
        main(data_id, model_id, pt_id)
        clear_hf_cache()