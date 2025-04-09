import matplotlib.pyplot as plt
import json
import numpy as np

# Configuration
RESULTS_PATH = '/home/monoshi/CodeSemantic/CodeSemantic/Results/block_results_10.json'
LINE_PLOT_PATH = 'Results/block_size_accuracy.png'
BOX_PLOT_PATH = 'Results/model_accuracy_distribution.png'

# Load results data
with open(RESULTS_PATH) as f:
    data = json.load(f)

# Get models and languages dynamically
MODELS_TO_PLOT = list(data.keys())
LANGUAGES = sorted({lang for model in MODELS_TO_PLOT 
                   if 'pt0' in data[model] 
                   for lang in data[model]['pt0'].keys()})

# Create color mapping for models
model_colors = plt.cm.tab10(np.linspace(0, 1, len(MODELS_TO_PLOT)))

# =============================================
# 1. LINE PLOTS (Accuracy by Block Size)
# =============================================
fig, axes = plt.subplots(len(LANGUAGES), 1, figsize=(12, 6 * len(LANGUAGES)))
if len(LANGUAGES) == 1:
    axes = [axes]  # Ensure axes is always a list

for lang_idx, language in enumerate(LANGUAGES):
    ax = axes[lang_idx]
    ax.set_title(f'{language.capitalize()} - Accuracy by Block Size', fontsize=14)
    ax.set_xlabel('Block Size', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Collect all block sizes across models
    all_block_sizes = set()
    for model in MODELS_TO_PLOT:
        if 'pt0' in data[model] and language in data[model]['pt0']:
            all_block_sizes.update(map(int, data[model]['pt0'][language]['block_results'].keys()))
    sorted_block_sizes = sorted(all_block_sizes)
    
    # Plot each model's accuracy by block size
    for model_idx, model_name in enumerate(MODELS_TO_PLOT):
        if 'pt0' not in data[model_name] or language not in data[model_name]['pt0']:
            continue
            
        model_data = data[model_name]['pt0'][language]
        block_results = model_data['block_results']
        
        x = []
        y = []
        for size in sorted_block_sizes:
            str_size = str(size)
            if str_size in block_results:
                x.append(size)
                y.append(block_results[str_size]['accuracy'])
        
        ax.plot(x, y, 'o-', color=model_colors[model_idx], 
                label=f'{model_name}', markersize=8, linewidth=2)

    ax.set_xticks(sorted_block_sizes)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_ylim(0, 1.05)

plt.tight_layout()
plt.savefig(LINE_PLOT_PATH, dpi=300, bbox_inches='tight')
plt.close()

# =============================================
# 2. BOX PLOTS (Overall Accuracy Distribution)
# =============================================
plt.figure(figsize=(12, 6))

# Prepare data for box plots
boxplot_data = []
model_labels = []

for model_idx, model_name in enumerate(MODELS_TO_PLOT):
    model_accuracies = []
    for language in LANGUAGES:
        if 'pt0' in data[model_name] and language in data[model_name]['pt0']:
            block_results = data[model_name]['pt0'][language]['block_results']
            # Weight accuracies by sample counts
            for size, results in block_results.items():
                model_accuracies.extend([results['accuracy']] * results['total'])
    
    if model_accuracies:  # Only include models with data
        boxplot_data.append(model_accuracies)
        model_labels.append(model_name)

# Create box plot with updated parameter name
box = plt.boxplot(boxplot_data, patch_artist=True, vert=True, 
                 tick_labels=model_labels, showmeans=True,
                 meanprops={'marker':'D', 'markerfacecolor':'red'})

# Color boxes to match line plots
for patch, color in zip(box['boxes'], model_colors[:len(boxplot_data)]):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

plt.title('Overall Accuracy Distribution Across Models', fontsize=14)
plt.ylabel('Accuracy', fontsize=12)
plt.grid(True, axis='y', alpha=0.3)
plt.ylim(0, 1.05)

# Rotate model names if needed
if len(MODELS_TO_PLOT) > 5:
    plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.savefig(BOX_PLOT_PATH, dpi=300, bbox_inches='tight')
plt.close()