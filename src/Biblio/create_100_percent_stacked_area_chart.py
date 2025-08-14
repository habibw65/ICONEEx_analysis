import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np

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
        'ground validation', 'field validation', 'field-based',
        'ground-based', 'field work', 'ground data'
    ]
    
    pattern = '|'.join(keywords)
    
    df['Category'] = df['Abstract'].apply(
        lambda x: 'RS + In-situ' if re.search(pattern, x) else 'RS only'
    )
    
    return df

def plot_100_percent_stacked_area_chart(df):
    df = df[(df['Year'] >= 2000) & (df['Year'] <= 2024)].copy()
    counts = df.groupby(['Year', 'Category']).size().unstack(fill_value=0)

    if 'RS only' not in counts.columns:
        counts['RS only'] = 0
    if 'RS + In-situ' not in counts.columns:
        counts['RS + In-situ'] = 0
        
    # Ensure the columns are in the desired stacking order
    counts = counts[['RS only', 'RS + In-situ']]

    # Calculate proportions
    total_pubs = counts.sum(axis=1)
    proportions = counts.div(total_pubs, axis=0) * 100

    plt.style.use('seaborn-v0_8-paper')
    fig, ax = plt.subplots(figsize=(14, 8))

    # Create the 100% stacked area chart
    stack_handles = ax.stackplot(proportions.index, proportions['RS only'], proportions['RS + In-situ'], 
                                 labels=['RS only', 'RS + In-situ'], 
                                 colors=['#0072B2', '#D55E00'], alpha=0.8)
    
    ax.set_xlabel('Year', fontsize=14)
    ax.set_ylabel('Proportion of Publications (%)', fontsize=14)
    ax.tick_params(axis='x', rotation=45, labelsize=12)
    ax.tick_params(axis='y', labelsize=12)

    # Get all handles and labels
    handles, labels = ax.get_legend_handles_labels()

    # Sort handles and labels based on the length of the labels
    legend_items = []
    for h, l in zip(handles, labels):
        legend_items.append((l, h))

    legend_items.sort(key=lambda x: len(x[0]))

    sorted_labels = [item[0] for item in legend_items]
    sorted_handles = [item[1] for item in legend_items]

    ax.legend(sorted_handles, sorted_labels, loc='upper left', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_ylim(0, 100) # Ensure y-axis goes from 0 to 100%
    ax.autoscale(axis='x', tight=True)

    plt.tight_layout()
    plt.savefig('figure_rs_trends_100_percent_stacked_area.png', dpi=300)

if __name__ == '__main__':
    ris_file = 'savedrecs.ris'
    papers_df = parse_ris(ris_file)
    categorized_df = categorize_papers(papers_df)
    plot_100_percent_stacked_area_chart(categorized_df)
