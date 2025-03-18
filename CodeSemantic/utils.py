import json
from datasets import load_dataset

from src.codellm import AbstLiteLLM, LocalVLLM
from src.pt import StatementPt1, StatementPt2, StatementPt3

SPLIT_SYM ="____SPLIT____"
def load_my_dataset(data_id):
    if data_id == 0:
        with open("dataset/statement_prediction_dataset.jsonl", 'r') as f:
            dataset = [json.loads(line) for line in f]

    else:
        raise NotImplementedError

    return dataset

def model_id2name_cls(model_id: int):
    API_MODEL_NUM = 3

    if model_id == 0:
        model_name = "gemini-1.5-flash-002"
        model_cls = AbstLiteLLM
        provider = "vertex_ai"
    elif model_id == 1:
        model_name = "gemini-2.0-flash-lite-preview-02-05"
        model_cls = AbstLiteLLM
        provider = "vertex_ai"
    elif model_id == 2:
        model_name = "anthropic.claude-3-5-haiku-20241022-v1:0"
        model_cls = AbstLiteLLM
        provider = "bedrock"
    elif model_id == 3:
        model_name = "anthropic.claude-3-5-sonnet-20241022-v2:0"
        model_cls = AbstLiteLLM
        provider = "bedrock"

    elif model_id == 4:
        model_name = "deepseek-ai/deepseek-coder-1.3b-instruct"
        model_cls = LocalVLLM
        provider = "openai"

    elif model_id == 5:
        model_name = "meta-llama/Llama-3.1-8B-Instruct"
        model_cls = LocalVLLM
        provider = "openai"

    elif model_id == 6:
        model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"
        model_cls = LocalVLLM
        provider = "openai"
    elif model_id == 7:
        model_name = "Qwen/Qwen2.5-7B-Instruct"
        model_cls = LocalVLLM
        provider = "openai"


    else:
        raise ValueError(f"Model ID {model_id} is not valid")
    is_lora = None
    chat_template_path = "chat_template/completation.jinjia"
    return provider, model_name, model_cls, is_lora, chat_template_path


def load_model(model_id):
    provider, model_name, model_cls, lora_path, chat_template_path = model_id2name_cls(model_id)
    model = model_cls(provider, model_name)

    model.model_name = model_name.split('/')[-1]
    return model

def load_pt(pt_id):
    if pt_id == 0:
        return StatementPt1( 'pt1', demos=[])
    elif pt_id == 1:
        return StatementPt2( 'pt2', demos=[])
    elif pt_id == 2:
        return StatementPt3( 'pt3', demos=[])
    else:
        raise ValueError(f"PT ID {pt_id} is not valid")

if __name__ == '__main__':

    ds = load_my_dataset(0)
    print()
