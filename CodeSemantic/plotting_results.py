import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_results(results_file='Results/all_results.json'):
    with open(results_file) as f:
        results = json.load(f)
    
    # Define the reasoning models we want to plot separately
    reasoning_models = {
        "DeepSeek-R1-Distill-Qwen-7B",
        "DeepSeek-R1-Distill-Llama-8B",
        "DeepSeek-R1-Distill-Qwen-14B"
    }
    
    data = []
    for model, pt_data in results.items():
        for pt, lang_data in pt_data.items():
            for lang, metrics in lang_data.items():
                data.append({
                    'Model': model,
                    'Language': lang,
                    'Accuracy': metrics['overall_accuracy'],
                    'Statement Type': 'Overall',
                    'Count': sum(metrics['type_counts'].values())
                })
                for stmt_type, acc in metrics['type_accuracy'].items():
                    data.append({
                        'Model': model,
                        'Language': lang,
                        'Accuracy': acc,
                        'Statement Type': stmt_type,
                        'Count': metrics['type_counts'][stmt_type]
                    })
    
    df = pd.DataFrame(data)
    
    # Filter for reasoning models
    reasoning_df = df[df['Model'].isin(reasoning_models)].copy()
    
    # Plot 1: Overall Accuracy for Reasoning Models
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 6), dpi=100)
    overall_reasoning_df = reasoning_df[reasoning_df['Statement Type'] == 'Overall'].copy()
    bar_plot = sns.barplot(data=overall_reasoning_df, x='Model', y='Accuracy', hue='Language')
    plt.title('Overall Accuracy - Reasoning Models', fontsize=14, pad=20)
    plt.ylim(0, 1)
    plt.legend(loc='upper right')
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.tick_params(axis='x', which='major', pad=10)
    plt.tight_layout()
    plt.savefig('Results/reasoning_models_overall_accuracy.png', bbox_inches='tight', dpi=300)
    plt.show()
    
    # Plot 2: Statement Type Accuracy for Python (Reasoning Models)
    plt.figure(figsize=(14, 6), dpi=100)
    python_reasoning_df = reasoning_df[(reasoning_df['Statement Type'] != 'Overall') & 
                                     (reasoning_df['Language'] == 'python')].copy()
    stmt_order = python_reasoning_df.groupby('Statement Type')['Accuracy'].mean().sort_values().index
    sns.barplot(data=python_reasoning_df, x='Statement Type', y='Accuracy', hue='Model', order=stmt_order)
    plt.title('Statement Type Accuracy - Python (Reasoning Models)', fontsize=14, pad=20)
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.tight_layout()
    plt.savefig('Results/reasoning_models_statement_accuracy_python.png', bbox_inches='tight', dpi=300)
    plt.show()
    
    # Plot 3: Statement Type Accuracy for C (Reasoning Models)
    plt.figure(figsize=(14, 6), dpi=100)
    c_reasoning_df = reasoning_df[(reasoning_df['Statement Type'] != 'Overall') & 
                                 (reasoning_df['Language'] == 'c')].copy()
    stmt_order = c_reasoning_df.groupby('Statement Type')['Accuracy'].mean().sort_values().index
    sns.barplot(data=c_reasoning_df, x='Statement Type', y='Accuracy', hue='Model', order=stmt_order)
    plt.title('Statement Type Accuracy - C (Reasoning Models)', fontsize=14, pad=20)
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.tight_layout()
    plt.savefig('Results/reasoning_models_statement_accuracy_c.png', bbox_inches='tight', dpi=300)
    plt.show()
    
    # Original plots for all models (unchanged from your original code)
    sns.set_theme(style="whitegrid")
    
    plt.figure(figsize=(12, 6), dpi=100)
    overall_df = df[df['Statement Type'] == 'Overall'].copy()
    bar_plot = sns.barplot(data=overall_df, x='Model', y='Accuracy', hue='Language')
    plt.title('Overall Accuracy by Model and Language', fontsize=14, pad=20)
    plt.ylim(0, 1)
    plt.legend(loc='upper right')
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.tick_params(axis='x', which='major', pad=10)
    plt.tight_layout()
    plt.savefig('Results/overall_accuracy.png', bbox_inches='tight', dpi=300)
    plt.show()
    
    plt.figure(figsize=(14, 6), dpi=100)
    python_df = df[(df['Statement Type'] != 'Overall') & (df['Language'] == 'python')].copy()
    stmt_order = python_df.groupby('Statement Type')['Accuracy'].mean().sort_values().index
    sns.barplot(data=python_df, x='Statement Type', y='Accuracy', hue='Model', order=stmt_order)
    plt.title('Statement Type Accuracy - Python', fontsize=14, pad=20)
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.tight_layout()
    plt.savefig('Results/statement_accuracy_python.png', bbox_inches='tight', dpi=300)
    plt.show()
    
    plt.figure(figsize=(14, 6), dpi=100)
    c_df = df[(df['Statement Type'] != 'Overall') & (df['Language'] == 'c')].copy()
    stmt_order = c_df.groupby('Statement Type')['Accuracy'].mean().sort_values().index
    sns.barplot(data=c_df, x='Statement Type', y='Accuracy', hue='Model', order=stmt_order)
    plt.title('Statement Type Accuracy - C', fontsize=14, pad=20)
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.tight_layout()
    plt.savefig('Results/statement_accuracy_c.png', bbox_inches='tight', dpi=300)
    plt.show()

if __name__ == '__main__':
    plot_results()