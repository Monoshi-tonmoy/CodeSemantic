import json
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patheffects as path_effects

# Load data
data = []
with open("/home/monoshi/CodeSemantic/CodeSemantic/statement_Accuracy_Results/statement_results.jsonl", "r") as f:
    for line in f:
        data.append(json.loads(line))

# Reorganize data structure
model_results = defaultdict(lambda: {
    "quantized": {
        "same": defaultdict(float),
        "different": defaultdict(float)
    },
    "non_quantized": {
        "same": defaultdict(float),
        "different": defaultdict(float)
    }
})

for entry in data:
    model = entry["Model"]
    quant = "quantized" if entry["quantization"] == "yes" else "non_quantized"
    incontext = entry["Incontext"]  
    shot = entry["shot"]
    accuracy = entry["accuracy"]
    
    model_results[model][quant][incontext][shot] = accuracy

models = list(model_results.keys())
colors = {'same': '#1f77b4', 'different': '#ff7f0e'}  

def create_incontext_comparison_plot(quant_mode):
    fig, ax = plt.subplots(figsize=(12, 6))
    

    shot = 3
    

    same_accuracies = [model_results[model][quant_mode]["same"][shot] for model in models]
    different_accuracies = [model_results[model][quant_mode]["different"][shot] for model in models]
    
    x = np.arange(len(models))
    width = 0.35
    

    bars_same = ax.bar(x - width/2, same_accuracies, width, 
                      label='Incontext: Same', color=colors['same'],
                      edgecolor='white', linewidth=0.5)
    bars_diff = ax.bar(x + width/2, different_accuracies, width, 
                      label='Incontext: Different', color=colors['different'],
                      edgecolor='white', linewidth=0.5)
    
    # # Add value labels on top of each bar
    # for bar in bars_same + bars_diff:
    #     height = bar.get_height()
    #     ax.annotate(f'{height:.2f}',
    #                 xy=(bar.get_x() + bar.get_width() / 2, height),
    #                 xytext=(0, 3),  # 3 points vertical offset
    #                 textcoords="offset points",
    #                 ha='center', va='bottom')
    
    # Customize plot
    if quant_mode == "non_quantized":
        display_mode = "Concrete"
    else:
        display_mode = "Quantized"
        
    ax.set_title(f'{display_mode} Value Accuracy Comparison (Shot=3)\nIncontext: Same vs Different', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', linestyle=':', alpha=0.7)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f'{display_mode}_value_incontext_comparison.png', bbox_inches='tight', dpi=300)
    plt.show()

# Create plots for both quantization modes
create_incontext_comparison_plot("quantized")
create_incontext_comparison_plot("non_quantized")