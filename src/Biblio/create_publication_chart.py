
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data
df = pd.read_csv('publications_by_topic_data.csv')

# Filter data up to 2024
df = df[df['Year'] <= 2024]

# Melt the dataframe to long format
df_melted = df.melt(id_vars='Year', var_name='Topic', value_name='Count')

# Calculate the total count for each topic and sort
topic_totals = df_melted.groupby('Topic')['Count'].sum().sort_values(ascending=False)
sorted_topics = sorted(topic_totals.index.tolist(), key=len)

# Define a colorblind-friendly color palette
colors = {
    'Fire': '#E69F00',      # Orange/Yellow
    'Hydrology': '#56B4E9', # Sky Blue
    'Vegetation': '#009E73',# Bluish Green
    'Land Use Land Cover': '#F0E442', # Yellow
    'Peat Extent/Mapping': '#0072B2', # Blue
    'Peat Subsidence': '#D55E00', # Vermillion
}

# Create the plot
plt.style.use('seaborn-v0_8-paper')
fig, ax = plt.subplots(figsize=(12, 8))

# Create the stacked bar chart
bottom = np.zeros(len(df['Year']))
for topic in sorted_topics:
    ax.bar(df['Year'], df[topic], bottom=bottom, label=topic, color=colors[topic])
    bottom += df[topic].values

# Set labels and title
ax.set_xlabel('Year')
ax.set_ylabel('Number of Publications')
ax.legend(title='Topic', loc='best')
ax.grid(True, which='both', linestyle='--', linewidth=0.5)

# Set x-axis ticks to show all years
ax.set_xticks(df['Year'])
ax.tick_params(axis='x', rotation=45)

# Save the figure
plt.savefig('publication_trends.png', dpi=300, bbox_inches='tight')
