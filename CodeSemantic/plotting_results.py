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
    plt.figure(figsize=(14, 10))
    

    plt.subplot(2, 1, 1)
    overall_df = df[df['Statement Type'] == 'Overall']
    sns.barplot(data=overall_df, x='Model', y='Accuracy', hue='Language')
    plt.title('Overall Accuracy by Model and Language')
    plt.ylim(0, 1)
    plt.legend(loc='upper right')
    
    plt.subplot(2, 1, 2)
    type_df = df[df['Statement Type'] != 'Overall']
    
    type_df['Model_Language'] = type_df['Model'] + ' (' + type_df['Language'] + ')'
    
    stmt_order = type_df.groupby('Statement Type')['Accuracy'].mean().sort_values().index
    
    sns.barplot(data=type_df, x='Statement Type', y='Accuracy', 
                hue='Model_Language', order=stmt_order)
    plt.title('Accuracy by Statement Type, Model and Language')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    plt.savefig('Results/model_comparison.png', bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    plot_results()