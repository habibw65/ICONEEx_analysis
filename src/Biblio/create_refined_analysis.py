import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np
from sklearn.linear_model import LinearRegression

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

def plot_refined_analysis(df):
    df = df[(df['Year'] >= 2000) & (df['Year'] <= 2024)].copy()
    counts = df.groupby(['Year', 'Category']).size().unstack(fill_value=0)

    if 'RS only' not in counts.columns:
        counts['RS only'] = 0
    if 'RS + In-situ' not in counts.columns:
        counts['RS + In-situ'] = 0

    # Calculate proportions
    total_pubs = counts.sum(axis=1)
    proportions = counts.div(total_pubs, axis=0) * 100

    plt.style.use('seaborn-v0_8-paper')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)

    # --- Top Plot: 100% Stacked Area Chart ---
    ax1.stackplot(proportions.index, proportions['RS only'], proportions['RS + In-situ'], 
                  labels=['RS only', 'RS + In-situ'], 
                  colors=['#0072B2', '#D55E00'], alpha=0.8)
    ax1.set_ylabel('Proportion of Publications (%)', fontsize=14)
    ax1.set_title('Shift in Research Methodology: RS Only vs. Integrated Approaches', fontsize=16, pad=20)
    ax1.legend(loc='upper left', fontsize=12)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax1.set_ylim(0, 100)

    # --- Bottom Plot: Line Chart with Trendline ---
    rs_in_situ_counts = counts['RS + In-situ']
    ax2.plot(rs_in_situ_counts.index, rs_in_situ_counts, label='RS + In-situ (Absolute Count)', color='#D55E00', marker='o', linestyle='-')

    # Calculate and plot trendline
    x = rs_in_situ_counts.index.values.reshape(-1, 1)
    y = rs_in_situ_counts.values
    model = LinearRegression()
    model.fit(x, y)
    trend = model.predict(x)
    r_squared = model.score(x, y)
    equation = f'y = {model.coef_[0]:.2f}x + {model.intercept_:.2f}'

    ax2.plot(rs_in_situ_counts.index, trend, color='black', linestyle='--', label=f'Trendline (R² = {r_squared:.2f})')
    
    ax2.set_xlabel('Year', fontsize=14)
    ax2.set_ylabel('Number of Publications', fontsize=14)
    ax2.set_title('Growth of Integrated \'RS + In-situ\' Studies', fontsize=16, pad=20)
    ax2.legend(fontsize=12)
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Set x-axis ticks for all years
    plt.xticks(df['Year'].unique(), rotation=45)
    ax2.tick_params(axis='x', labelsize=12)
    ax1.tick_params(axis='y', labelsize=12)
    ax2.tick_params(axis='y', labelsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig('figure_rs_trends_refined.png', dpi=300)

if __name__ == '__main__':
    ris_file = 'savedrecs.ris'
    papers_df = parse_ris(ris_file)
    categorized_df = categorize_papers(papers_df)
    plot_refined_analysis(categorized_df)
