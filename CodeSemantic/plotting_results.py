import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_results(results_file='Results/all_results.json'):
    with open(results_file) as f:
        results = json.load(f)
    
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