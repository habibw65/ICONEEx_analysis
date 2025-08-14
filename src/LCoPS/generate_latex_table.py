

import pandas as pd
import os

output_dir = '/Users/habibw/Documents/LCoPS/'
output_latex_path = os.path.join(output_dir, 'landcover_proportions_table.tex')

try:
    # Define the data for the LaTeX table directly from provided percentages
    data = {
        'Land cover': ['Grasslands', 'Forests', 'Near-natural peatlands', 'Heath', 'Cutover', 'Others', 'Bare peat', 'Scrub', 'Hedgerows', 'Bracken'],
        'Proportion on all peat soils': [30.6, 20.1, 16.2, 15.0, 6.4, 4.9, 3.1, 2.2, 1.4, 0.0],
        'Proportion on shallow peat-peaty soils': [37.1, 18.7, 4.6, 26.3, 0.0, 4.9, 0.0, 4.0, 1.5, 2.9]
    }
    df_table = pd.DataFrame(data)

    # Generate LaTeX table string
    # Use triple quotes for multiline string and double backslashes for LaTeX commands
    latex_table = """
\begin{table}[h!]
\centering
\caption{Land Cover Proportions on Peat Soils}
\label{tab:land_cover_proportions}
\begin{tabular}{l c c}
\toprule
Land cover & Proportion on all & Proportion on shallow \\
           & peat soils (%) & peat-peaty soils (%) \\
\midrule
"""

    for index, row in df_table.iterrows():
        # Construct the row string using string formatting and explicit newlines
        # Use four backslashes to produce two backslashes in the final LaTeX output (for \\)
        row_str = "{} & {} & {} \\\\n".format(row['Land cover'], row['Proportion on all peat soils'], row['Proportion on shallow peat-peaty soils'])
        latex_table += row_str

    latex_table += """
\bottomrule
\end{tabular}
\end{table}
"""

    with open(output_latex_path, 'w') as f:
        f.write(latex_table)
    
    print(f"LaTeX table saved to: {output_latex_path}")

except Exception as e:
    print(f"An error occurred: {e}")

