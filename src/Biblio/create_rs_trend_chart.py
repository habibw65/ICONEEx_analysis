import pandas as pd
import matplotlib.pyplot as plt
import re

def parse_ris(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    papers = []
    records = content.split('ER  -')
    
    for record in records:
        if not record.strip():
            continue
            
        year = None
        abstract = None
        
        year_match = re.search(r'\nPY  - (\d{4})', record)
        if year_match:
            year = int(year_match.group(1))
            
        abstract_match = re.search(r'\nAB  - (.*?)(?=\n[A-Z]{2}  - |$)', record, re.DOTALL)
        if abstract_match:
            abstract = abstract_match.group(1).strip().lower()
            
        if year and abstract:
            papers.append({'Year': year, 'Abstract': abstract})
            
    return pd.DataFrame(papers)

def categorize_papers(df):
    keywords = [
        'in-situ', 'in situ', 'ground truth', 'field data', 
        'field survey', 'field measurements', 'ground check', 
        'ground validation', 'field validation'
    ]
    
    pattern = '|'.join(keywords)
    
    df['Category'] = df['Abstract'].apply(
        lambda x: 'RS + In-situ' if re.search(pattern, x) else 'RS only'
    )
    
    return df

def plot_trends(df):
    df = df[(df['Year'] >= 2000) & (df['Year'] <= 2024)]
    counts = df.groupby(['Year', 'Category']).size().unstack(fill_value=0)
    
    # Ensure both categories are present
    if 'RS only' not in counts.columns:
        counts['RS only'] = 0
    if 'RS + In-situ' not in counts.columns:
        counts['RS + In-situ'] = 0
        
    counts = counts.sort_index()

    plt.style.use('seaborn-v0_8-paper')
    fig, ax = plt.subplots(figsize=(14, 8))
    
    ax.plot(counts.index, counts['RS only'], label='RS only', color='#0072B2', marker='o')
    ax.plot(counts.index, counts['RS + In-situ'], label='RS + In-situ', color='#D55E00', marker='o')
    
    ax.set_xlabel('Year', fontsize=14)
    ax.set_ylabel('Number of Publications', fontsize=14)
    ax.tick_params(axis='x', rotation=45, labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig('figure_rs_only_vs_combo_trend.png', dpi=300)

    # Save counts to CSV
    counts.to_csv('rs_trends_yearly_counts.csv')

if __name__ == '__main__':
    ris_file = 'savedrecs.ris'
    papers_df = parse_ris(ris_file)
    categorized_df = categorize_papers(papers_df)
    plot_trends(categorized_df)
