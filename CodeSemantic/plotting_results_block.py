import matplotlib.pyplot as plt
import json
import numpy as np

# Configuration
RESULTS_PATH = '/home/monoshi/CodeSemantic/CodeSemantic/Results/block_results.json'
OUTPUT_PATH = 'Results/multi_model_block_accuracy_comparison.png'

# Load results data
with open(RESULTS_PATH) as f:
    data = json.load(f)

# Dynamically get all model names and languages
MODELS_TO_PLOT = list(data.keys())
LANGUAGES = set()
for model in MODELS_TO_PLOT:
    if 'pt0' in data[model]:
        LANGUAGES.update(data[model]['pt0'].keys())
LANGUAGES = sorted(LANGUAGES)

# Set up plotting
fig, axes = plt.subplots(len(LANGUAGES), 1, figsize=(12, 8 * len(LANGUAGES)))
if len(LANGUAGES) == 1:
    axes = [axes]  # Ensure axes is always a list
fig.suptitle('Accuracy by Block Size Across Models', fontsize=16, y=1.02)

# Color palette for models
model_colors = plt.cm.tab10(np.linspace(0, 1, len(MODELS_TO_PLOT)))

for lang_idx, language in enumerate(LANGUAGES):
    ax = axes[lang_idx]
    ax.set_title(f'{language.capitalize()}', fontsize=14)
    ax.set_xlabel('Block Size', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Collect all block sizes across models for consistent x-axis
    all_block_sizes = set()
    for model in MODELS_TO_PLOT:
        if 'pt0' in data[model] and language in data[model]['pt0']:
            all_block_sizes.update(map(int, data[model]['pt0'][language]['block_results'].keys()))
    sorted_block_sizes = sorted(all_block_sizes)
    
    # Plot each model
    for model_idx, model_name in enumerate(MODELS_TO_PLOT):
        if 'pt0' not in data[model_name] or language not in data[model_name]['pt0']:
            continue
            
        model_data = data[model_name]['pt0'][language]
        block_results = model_data['block_results']
        
        # Prepare data for plotting
        x = []
        y = []
        counts = []
        for size in sorted_block_sizes:
            str_size = str(size)
            if str_size in block_results:
                x.append(size)
                y.append(block_results[str_size]['accuracy'])
                counts.append(block_results[str_size]['total'])
            else:
                x.append(size)
                y.append(None)  # Will appear as gap in plot
                counts.append(0)
        
        # Plot the line with markers
        line, = ax.plot(x, y, 'o-', color=model_colors[model_idx], 
                       label=f'{model_name} (Avg: {model_data["overall_accuracy"]:.2f})',
                       markersize=8, linewidth=2)
        
        # Add count annotations
        for size, acc, count in zip(x, y, counts):
            if acc is not None and count > 0:
                ax.annotate(f'n={count}', (size, acc), 
                           textcoords="offset points", xytext=(0,8), 
                           ha='center', fontsize=8, color=model_colors[model_idx])

    # Finalize subplot
    ax.set_xticks(sorted_block_sizes)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1.05)  # Fixed y-axis for accuracy comparison

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
plt.show()