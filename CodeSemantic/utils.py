import json
from collections import defaultdict
from datasets import load_dataset
from src.codellm import AbstLiteLLM, LocalVLLM
from src.pt import StatementPt1, StatementPt2, StatementPt3
import os

def save_jsonl(data, filename):
    with open(filename, 'w') as f:
        for entry in data:
            serialized_entry = serialize_vllm_objects(entry)
            json.dump(serialized_entry, f)
            f.write('\n')

def load_existing_results(filename='all_results.json'):
    """Load existing results if file exists, otherwise return empty dict"""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def save_results_to_json(model_name, pt_id, language, overall_accuracy, 
                        type_accuracy, detailed_results, 
                        filename='all_results.json', is_block_based=False):
    """Save results to JSON in Results folder, preserving existing data"""
    os.makedirs('Results', exist_ok=True)
    
    if is_block_based:
        filename = 'block_results.json'
    
    results_path = os.path.join('Results', filename)
    
    existing_results = load_existing_results(results_path)
    
    if model_name not in existing_results:
        existing_results[model_name] = {}
    if f'pt{pt_id}' not in existing_results[model_name]:
        existing_results[model_name][f'pt{pt_id}'] = {}
    
    result_entry = {
        'overall_accuracy': overall_accuracy,
        'is_block_based': is_block_based
    }
    
    if is_block_based:
        # detailed_results already contains the properly structured data
        result_entry.update({
            'block_results': detailed_results,
            'sample_counts': {
                size: results['total']
                for size, results in detailed_results.items()
            }
        })
    else:
        result_entry.update({
            'type_accuracy': type_accuracy,
            'type_counts': detailed_results
        })
    
    existing_results[model_name][f'pt{pt_id}'][language] = result_entry
    
    with open(results_path, 'w') as f:
        json.dump(existing_results, f, indent=2)

def load_results_for_model(model_name, is_block_based=False):
    """Load results for a specific model from the appropriate file"""
    filename = 'block_results.json' if is_block_based else 'all_results.json'
    results_path = os.path.join('Results', filename)
    
    try:
        with open(results_path, 'r') as f:
            all_results = json.load(f)
            return all_results.get(model_name, {})
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def get_available_models(is_block_based=False):
    """Get list of models that have results in the specified file"""
    filename = 'block_results.json' if is_block_based else 'all_results.json'
    results_path = os.path.join('Results', filename)
    
    try:
        with open(results_path, 'r') as f:
            all_results = json.load(f)
            return list(all_results.keys())
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def serialize_vllm_objects(obj):
    """Recursively convert VLLM objects to serializable formats"""
    if isinstance(obj, (str, int, float, bool)):
        return obj
    elif isinstance(obj, dict):
        return {key: serialize_vllm_objects(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [serialize_vllm_objects(item) for item in obj]
    elif hasattr(obj, '__dict__'):
        return serialize_vllm_objects(obj.__dict__)
    else:
        return str(obj)

def serialize_response(response):
    """Convert non-serializable objects in the response to serializable formats"""
    serialized = {}
    for key, value in response.items():
        if key == 'model_pred':
            if hasattr(value, '__dict__'):
                serialized[key] = value.__dict__
            else:
                serialized[key] = str(value)
        else:
            serialized[key] = value
    return serialized

SPLIT_SYM = "____SPLIT____"

def load_my_dataset(data_id):
    if data_id == 0:
        with open("dataset/statement_prediction_dataset.jsonl", 'r') as f:
            dataset = [json.loads(line) for line in f]
    elif data_id == 1:
        with open("dataset/statement_prediction_dataset_C.jsonl", 'r') as f:
            dataset = [json.loads(line) for line in f]
    elif data_id == 2:
        with open("dataset/incremental_statement_prediction_python.jsonl", 'r') as f:
            dataset = [json.loads(line) for line in f]
    elif data_id == 3:
        with open("dataset/incremental_statement_prediction_c.jsonl", 'r') as f:
            dataset = [json.loads(line) for line in f]
    else:
        raise NotImplementedError
    return dataset

# Model loading
def model_id2name_cls(model_id: int):
    model_map = {
        0: ("gemini-1.5-flash-002", AbstLiteLLM, "vertex_ai"),
        1: ("gemini-2.0-flash-lite-preview-02-05", AbstLiteLLM, "vertex_ai"),
        2: ("anthropic.claude-3-5-haiku-20241022-v1:0", AbstLiteLLM, "bedrock"),
        3: ("anthropic.claude-3-5-sonnet-20241022-v2:0", AbstLiteLLM, "bedrock"),
        4: ("deepseek-ai/deepseek-coder-1.3b-instruct", LocalVLLM, "openai"),
        5: ("Qwen/Qwen2.5-7B-Instruct", LocalVLLM, "openai"),
        6: ("microsoft/Phi-3-medium-128k-instruct", LocalVLLM, "openai"),
        7: ("meta-llama/Llama-3.1-8B-Instruct", LocalVLLM, "openai"),
        8: ("Qwen/Qwen2.5-14B-Instruct-1M", LocalVLLM, "openai"),
        9: ("Qwen/Qwen2.5-Coder-7B-Instruct", LocalVLLM, "openai"),
        10: ("deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct", LocalVLLM, "openai"),
        11: ("microsoft/Phi-4-mini-instruct", LocalVLLM, "openai"),
        12: ("microsoft/Phi-3.5-mini-instruct", LocalVLLM, "openai"),
        13: ("ibm-granite/granite-3.2-8b-instruct", LocalVLLM, "openai"),
        14: ("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", LocalVLLM, "openai"),
    }
    
    if model_id not in model_map:
        raise ValueError(f"Model ID {model_id} is not valid")
        
    model_name, model_cls, provider = model_map[model_id]
    return provider, model_name, model_cls, None, "chat_template/completation.jinjia"

def load_model(model_id):
    provider, model_name, model_cls, lora_path, chat_template_path = model_id2name_cls(model_id)
    model = model_cls(provider, model_name)
    model.model_name = model_name.split('/')[-1]
    return model

def load_pt(pt_id):
    pt_map = {
        0: StatementPt1('pt1', demos=[]),
        1: StatementPt2('pt2', demos=[]),
        2: StatementPt3('pt3', demos=[])
    }
    
    if pt_id not in pt_map:
        raise ValueError(f"PT ID {pt_id} is not valid")
    return pt_map[pt_id]


def get_default_config():
    return {
        'temperature': 0.8,
        "top_p": 0.95,
        "max_tokens": 1024,
        "tp_size": 1,
        "dtype": "float16",
        "stop": [
            "\n>>>", "\n$", '\nclass',
            '\ndef', '\n#', '\nprint',
            "\n@", "\nif __name__ == '__main__':"
        ]
    }