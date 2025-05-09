import json
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patheffects as path_effects


data = []
with open("/home/monoshi/CodeSemantic/CodeSemantic/statement_Accuracy_Results/statement_results.jsonl", "r") as f:
    for line in f:
        data.append(json.loads(line))

model_results = defaultdict(lambda: {
    "quantized": defaultdict(float),
    "non_quantized": defaultdict(float)
})

for entry in data:
    model = entry["Model"]
    quant = "quantized" if entry["quantization"] == "yes" else "non_quantized"
    shot = entry["shot"]
    accuracy = entry["accuracy"]
    model_results[model][quant][shot] = accuracy

models = list(model_results.keys())
shots = [0, 1, 2, 3]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

def get_best_shots(model_results, quant_mode):
    best_shots = {}
    for model in model_results:
        shot_accuracies = model_results[model][quant_mode]
        best_shot = max(shot_accuracies.items(), key=lambda x: x[1])[0]
        best_shots[model] = best_shot
    return best_shots

def create_plot(quant_mode, title_suffix):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, shot in enumerate(shots):
        accuracies = [model_results[model][quant_mode][shot] for model in models]
        bars = ax.bar(
            np.arange(len(models)) + i * 0.2, accuracies,
            width=0.2, color=colors[i], label=f'Shot {shot}',
            edgecolor='white', linewidth=0.5
        )
    
    if quant_mode == "non_quantized":
        quant_mode = "concrete"

    ax.set_title(f'{quant_mode.capitalize()} Statement Prediction Accuracy', pad=20)
    ax.set_xticks(np.arange(len(models)) + 0.3)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', linestyle=':', alpha=0.7)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    
    plt.savefig(f'{quant_mode}_models_vs_shots.png', bbox_inches='tight', dpi=300)
    plt.show()

create_plot("quantized", "Yes")
create_plot("non_quantized", "No")