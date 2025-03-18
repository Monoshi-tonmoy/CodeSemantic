import numpy as np
from utils import load_my_dataset
from utils import load_model
from utils import load_pt


# def save_to_file(model_name, pt_id, accuracy):
#  
#     with open('results.txt', 'a') as f:
#         f.write(f"Model Name: {model_name}, PT ID: {pt_id}, Accuracy: {accuracy}\n")


def main(data_id, model_id, pt_id):
    config = {
        'temperature': 0.8,
        "top_p": 0.95,
        "max_tokens": 1024,  # args.max_tokens,
        "tp_size": 1,  # args.tp_size,
        "dtype": "float16",
        "stop": [
            "\n>>>", "\n$", '\nclass',
            '\ndef', '\n#', '\nprint',
            "\n@", "\nif __name__ == '__main__':"
        ]
    }

    dataset = load_my_dataset(data_id)
    dataset = dataset[:100]
    model = load_model(model_id)
    model.init_ai_kwargs(config)
    pt = load_pt(pt_id)
    
    
    model_name = model.model_name
    
    res = model.chat_batch(pt, dataset)
    is_correct = []
    for d in res:
        try:
            # Checking if 'pred_ans' exists and is not empty 
            if 'pred_ans' in d and len(d['pred_ans']) > 0:
                is_correct.append(d['ori_task']['Value After Statement Execution'] == d['pred_ans'][0])
            else:
            #if empty, incorrect
                is_correct.append(False)
                print(f"Warning: Missing or empty 'pred_ans' in response for model {model_name}, PT ID {pt_id}")
        except KeyError:
            # Handle cases where 'ori_task' or 'Value After Statement Execution' is missing
            is_correct.append(False)
            print(f"Warning: Missing 'ori_task' or 'Value After Statement Execution' in response for model {model_name}, PT ID {pt_id}")

    accuracy = np.mean(is_correct)

    print(f"Here goes the output: {model_name}, {pt_id}, {accuracy}")

    save_to_file(model_name, pt_id, accuracy)


if __name__ == '__main__':
    main(0, 6, 0)
    main(0, 6, 1)
    main(0, 6, 2)
    # main(0, 0, 0)
    # main(0, 0, 1)
    # main(0, 0, 2)

    # main(0, 2, 0)
    # main(0, 2, 1)
    # main(0, 2, 2)