import json
import matplotlib.pyplot as plt
import numpy as np

def plot_models_vs_shots(json_file_path, save_path=None):
    with open(json_file_path) as f:
        data = json.load(f)
    
    models = []
    shot_0_acc = []
    shot_1_acc = []
    shot_2_acc = []
    
    for model_name, model_data in data.items():
        models.append(model_name)
        statement_data = model_data['pt0']['python']['statement']
        
        shot_0_acc.append(statement_data['shot0']['overall_accuracy'])
        
        shot_1_acc.append(
            statement_data['shot1']['CoT_no']['Incontext_different']['overall_accuracy']
        )

        shot_2_acc.append(
            statement_data['shot2']['CoT_no']['Incontext_different']['overall_accuracy']
        )
    
    shot_0_acc = [x * 100 for x in shot_0_acc]
    shot_1_acc = [x * 100 for x in shot_1_acc]
    shot_2_acc = [x * 100 for x in shot_2_acc]
    

    x = np.arange(len(models))  
    width = 0.25  
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    rects1 = ax.bar(x - width, shot_0_acc, width, label='0-shot', color='#1f77b4')
    rects2 = ax.bar(x, shot_1_acc, width, label='1-shot', color='#ff7f0e')
    rects3 = ax.bar(x + width, shot_2_acc, width, label='2-shot', color='#2ca02c')
    

    ax.set_ylabel('Overall Accuracy (%)')
    ax.set_title('Model Performance by Number of Shots')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    

    # def autolabel(rects):
    #     for rect in rects:
    #         height = rect.get_height()
    #         ax.annotate(f'{height:.1f}%',
    #                     xy=(rect.get_x() + rect.get_width() / 2, height),
    #                     xytext=(0, 3),  
    #                     textcoords="offset points",
    #                     ha='center', va='bottom', fontsize=9)
    
    # autolabel(rects1)
    # autolabel(rects2)
    # autolabel(rects3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
def plot_shot_comparison(json_file_path, save_path=None):
    """
    Compare 1-shot vs 2-shot performance for both CoT and Incontext configurations
    (8 bars per model).
    """
    with open(json_file_path) as f:
        data = json.load(f)

    models = []
    cot_yes_1shot_d = []
    cot_yes_1shot_s = []
    cot_no_1shot_d = []
    cot_no_1shot_s = []
    cot_yes_2shot_d = []
    cot_yes_2shot_s = []
    cot_no_2shot_d = []
    cot_no_2shot_s = []

    for model_name, model_data in data.items():
        models.append(model_name)
        statement_data = model_data['pt0']['python']['statement']

        cot_yes_1shot_d.append(statement_data['shot1']['CoT_yes']['Incontext_different']['overall_accuracy'])
        cot_yes_1shot_s.append(statement_data['shot1']['CoT_yes']['Incontext_same']['overall_accuracy'])
        cot_no_1shot_d.append(statement_data['shot1']['CoT_no']['Incontext_different']['overall_accuracy'])
        cot_no_1shot_s.append(statement_data['shot1']['CoT_no']['Incontext_same']['overall_accuracy'])
        cot_yes_2shot_d.append(statement_data['shot2']['CoT_yes']['Incontext_different']['overall_accuracy'])
        cot_yes_2shot_s.append(statement_data['shot2']['CoT_yes']['Incontext_same']['overall_accuracy'])
        cot_no_2shot_d.append(statement_data['shot2']['CoT_no']['Incontext_different']['overall_accuracy'])
        cot_no_2shot_s.append(statement_data['shot2']['CoT_no']['Incontext_same']['overall_accuracy'])

    # Convert to percentage
    def to_percent(lst): return [x * 100 for x in lst]
    cot_yes_1shot_d = to_percent(cot_yes_1shot_d)
    cot_yes_1shot_s = to_percent(cot_yes_1shot_s)
    cot_no_1shot_d = to_percent(cot_no_1shot_d)
    cot_no_1shot_s = to_percent(cot_no_1shot_s)
    cot_yes_2shot_d = to_percent(cot_yes_2shot_d)
    cot_yes_2shot_s = to_percent(cot_yes_2shot_s)
    cot_no_2shot_d = to_percent(cot_no_2shot_d)
    cot_no_2shot_s = to_percent(cot_no_2shot_s)

    x = np.arange(len(models))
    width = 0.1

    fig, ax = plt.subplots(figsize=(16, 8))

    rects = []
    rects.append(ax.bar(x - 3.5 * width, cot_yes_1shot_d, width, label='1-shot | CoT=yes | Inctx=dif', color='#1f77b4'))
    rects.append(ax.bar(x - 2.5 * width, cot_yes_1shot_s, width, label='1-shot | CoT=yes | Inctx=same', color='#aec7e8'))
    rects.append(ax.bar(x - 1.5 * width, cot_no_1shot_d, width, label='1-shot | CoT=no | Inctx=dif', color='#7f7f7f'))
    rects.append(ax.bar(x - 0.5 * width, cot_no_1shot_s, width, label='1-shot | CoT=no | Inctx=same', color='#c7c7c7'))
    rects.append(ax.bar(x + 0.5 * width, cot_yes_2shot_d, width, label='2-shot | CoT=yes | Inctx=dif', color='#ff7f0e'))
    rects.append(ax.bar(x + 1.5 * width, cot_yes_2shot_s, width, label='2-shot | CoT=yes | Inctx=same', color='#ffbb78'))
    rects.append(ax.bar(x + 2.5 * width, cot_no_2shot_d, width, label='2-shot | CoT=no | Inctx=dif', color='#2ca02c'))
    rects.append(ax.bar(x + 3.5 * width, cot_no_2shot_s, width, label='2-shot | CoT=no | Inctx=same', color='#98df8a'))

    ax.set_ylabel('Overall Accuracy (%)')
    ax.set_title('1-shot vs 2-shot Performance Comparison (All CoT & Incontext Combinations)')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_ylim(0, 100)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')

    # def autolabel(rect_group):
    #     for rect in rect_group:
    #         height = rect.get_height()
    #         ax.annotate(f'{height:.1f}%',
    #                     xy=(rect.get_x() + rect.get_width() / 2, height),
    #                     xytext=(0, 2),
    #                     textcoords="offset points",
    #                     ha='center', va='bottom', fontsize=8)

    # for rect_group in rects:
    #     autolabel(rect_group)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
def plot_type_accuracy_comparison(json_file_path, save_path=None):
    """
    Plot type accuracy comparison for 0-shot, 1-shot and 2-shot with CoT_yes and CoT_no.
    Handles single or multiple models correctly.
    """
    with open(json_file_path) as f:
        data = json.load(f)
    
    models = list(data.keys())
    statement_types = ["Assignment", "Branch", "API", "Arithmetic Assignment", "Constant Assignment"]
    
    # Determine grid layout
    ncols = 2
    nrows = (len(models) + ncols - 1) // ncols
    
    # Adjust figure size based on number of rows
    fig_height = 6 * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, fig_height))
    
    # If only one model, wrap axes in a list
    if len(models) == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('Type Accuracy Comparison (Including Zero-shot)', y=1.02, fontsize=14)
    
    for i, model_name in enumerate(models):
        row = i // ncols
        col = i % ncols
        ax = axes[row, col]
        
        model_data = data[model_name]
        statement_data = model_data['pt0']['python']['statement']
        
        # Get the data for each configuration
        zero_shot = statement_data['shot0']['type_accuracy']
        cot_yes_1shot = statement_data['shot1']['CoT_yes']['Incontext_different']['type_accuracy']
        cot_no_1shot = statement_data['shot1']['CoT_no']['Incontext_different']['type_accuracy']
        cot_yes_2shot = statement_data['shot2']['CoT_yes']['Incontext_different']['type_accuracy']
        cot_no_2shot = statement_data['shot2']['CoT_no']['Incontext_different']['type_accuracy']
        
        # Convert to percentages
        zero_shot = [zero_shot[typ] * 100 for typ in statement_types]
        cot_yes_1shot = [cot_yes_1shot[typ] * 100 for typ in statement_types]
        cot_no_1shot = [cot_no_1shot[typ] * 100 for typ in statement_types]
        cot_yes_2shot = [cot_yes_2shot[typ] * 100 for typ in statement_types]
        cot_no_2shot = [cot_no_2shot[typ] * 100 for typ in statement_types]
        
        x = np.arange(len(statement_types))
        width = 0.15
        
        bars = []
        bars.append(ax.bar(x - 2*width, zero_shot, width, label='0-shot', color='#7f7f7f'))
        bars.append(ax.bar(x - width, cot_yes_1shot, width, label='1-shot CoT=yes', color='#1f77b4'))
        bars.append(ax.bar(x, cot_no_1shot, width, label='1-shot CoT=no', color='#aec7e8'))
        bars.append(ax.bar(x + width, cot_yes_2shot, width, label='2-shot CoT=yes', color='#ff7f0e'))
        bars.append(ax.bar(x + 2*width, cot_no_2shot, width, label='2-shot CoT=no', color='#ffbb78'))
        
        ax.set_title(model_name, pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(statement_types, rotation=45, ha='right')
        ax.set_ylabel('Accuracy (%)')
        ax.set_ylim(0, 100)
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels on top of bars if there's space
        # if len(models) <= 4:
        #     for bar_group in bars:
        #         for bar in bar_group:
        #             height = bar.get_height()
        #             ax.annotate(f'{height:.1f}',
        #                         xy=(bar.get_x() + bar.get_width()/2, height),
        #                         xytext=(0, 3),
        #                         textcoords="offset points",
        #                         ha='center', va='bottom', fontsize=8)
    
    # Hide any empty subplots
    for i in range(len(models), nrows * ncols):
        row = i // ncols
        col = i % ncols
        axes[row, col].axis('off')
    
    # Create a unified legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=5,bbox_to_anchor=(0.5, 1.05))
    
    plt.tight_layout(pad=3.0)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def main():
    plot_models_vs_shots('statement_predictions.json', 'models_vs_shots.png')
    plot_shot_comparison('statement_predictions.json', save_path='shot_comparison.png')
    plot_type_accuracy_comparison('statement_predictions.json', 'type_accuracy_comparison.png')

    

if __name__ == "__main__":
    main()
