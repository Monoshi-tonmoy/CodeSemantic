import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# Define reasoning models
REASONING_MODELS = {
    "DeepSeek-R1-Distill-Qwen-7B",
    "DeepSeek-R1-Distill-Llama-8B",
    "DeepSeek-R1-Distill-Qwen-14B",
    "granite-3.2-8b-instruct",
    "granite-3.2-8b-instruct-preview",
}

# Load and process data
data = []
with open("/home/monoshi/CodeSemantic/CodeSemantic/loop_Accuracy_Results/loop_python_results.jsonl", "r") as f:
    for line in f:
        data.append(json.loads(line))

# Process data into DataFrame (non-quantized, zero-shot only)
loop_data = []
for record in data:
    if record.get("quantization") == "no" and record.get("shot") == 0:
        loop_data.append({
            "Model": record["Model"],
            "Accuracy": record["accuracy"],
            "Settings": record["settings"],
            "IsReasoning": record["Model"] in REASONING_MODELS
        })

df = pd.DataFrame(loop_data)

# Sort models: non-reasoning first, then reasoning
models_sorted = sorted(df['Model'].unique(), key=lambda x: x in REASONING_MODELS)

# Assign categorical order
df['Model'] = pd.Categorical(df['Model'], categories=models_sorted, ordered=True)
df = df.sort_values('Model')

# Plotting
settings = df['Settings'].unique()
n_models = len(models_sorted)
n_settings = len(settings)
x = np.arange(n_models)  # the label locations
width = 0.2  # width of each bar

# Color palette for settings
setting_colors = plt.get_cmap('Set2').colors[:n_settings]

fig, ax = plt.subplots(figsize=(14, 7))

for i, setting in enumerate(settings):
    setting_df = df[df['Settings'] == setting].set_index('Model').reindex(models_sorted).reset_index()
    
    # Calculate bar positions with offsets
    offset = (i - (n_settings-1)/2) * width
    bars = ax.bar(x + offset, setting_df['Accuracy'], width, label=setting, 
                  color=setting_colors[i],
                  edgecolor=['red' if r else 'black' for r in setting_df['IsReasoning']],
                  hatch=['///' if r else None for r in setting_df['IsReasoning']])
    
    # # Add value labels
    # for bar in bars:
    #     height = bar.get_height()
    #     ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
    #             f'{height:.2f}', ha='center', va='bottom', fontsize=9)


legend_elements = []

for i, setting in enumerate(settings):
    if setting == "body":
        setting = "In-loop value"     
    elif setting == "after":
        setting = "Post-loop value"     
    legend_elements.append(Patch(facecolor=setting_colors[i], label=setting))
    

legend_elements.extend([
    Patch(facecolor='white', edgecolor='red', hatch='///', label='Reasoning Models'),
    Patch(facecolor='white', edgecolor='black', label='Non-Reasoning Models')
])


# Configure plot appearance
ax.set_title('Loop Prediction Accuracy across Settings', pad=20, fontsize=16)
ax.set_xlabel('Model', labelpad=10)
ax.set_ylabel('Accuracy', labelpad=10)
ax.set_xticks(x)
ax.set_xticklabels(models_sorted, rotation=45, ha='right')
ax.set_ylim(0, 1.1)
ax.grid(axis='y', linestyle='--', alpha=0.3)
ax.legend(handles=legend_elements, bbox_to_anchor=(1.02, 1), loc='upper left')

plt.tight_layout()
plt.savefig('loop_accuracy_combined.png', dpi=300, bbox_inches='tight')
plt.show()
