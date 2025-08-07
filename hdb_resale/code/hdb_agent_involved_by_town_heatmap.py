import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# Paths to the datasets (update as needed)
CEA_CSV = '../data/CEA_SalespersonsPropertyTransactionRecordsresidential.csv'  # CEA agent transactions
HDB_CSV = '../data/hdb_resale_flat_prices.csv'  # All HDB resale transactions

# Load datasets
df_cea = pd.read_csv(CEA_CSV)
df_hdb = pd.read_csv(HDB_CSV)

# --- Preprocess CEA data ---
df_cea = df_cea[df_cea['property_type'].str.upper() == 'HDB']
df_cea = df_cea[df_cea['transaction_type'].str.upper() == 'RESALE']
df_cea = df_cea[df_cea['transaction_date'].notnull()]
df_cea['year'] = pd.to_datetime(df_cea['transaction_date'], format='%b-%Y', errors='coerce').dt.year

# Only keep valid years
df_cea = df_cea[df_cea['year'].notnull()]
df_cea['year'] = df_cea['year'].astype(int)

# --- Preprocess HDB data ---
df_hdb['month'] = pd.to_datetime(df_hdb['month'], format='%Y-%m')
df_hdb['year'] = df_hdb['month'].dt.year

# Filter for years 2017-2024
years = list(range(2017, 2025))
df_cea = df_cea[df_cea['year'].isin(years)]
df_hdb = df_hdb[df_hdb['year'].isin(years)]

towns = sorted(df_hdb['town'].unique())

# Compute agent involvement rate by town and year
results = []
for year in years:
    for town in towns:
        total = df_hdb[(df_hdb['year'] == year) & (df_hdb['town'] == town)].shape[0]
        agent = df_cea[(df_cea['year'] == year) & (df_cea['town'] == town)].shape[0]
        pct_agent = (agent / total * 100) if total > 0 else np.nan
        results.append({'Year': year, 'Town': town, '% Agent-Involved': round(pct_agent, 2)})

df_heat = pd.DataFrame(results)

# Pivot for heatmap
heatmap_data = df_heat.pivot(index='Town', columns='Year', values='% Agent-Involved')

# Plotly interactive heatmap
fig = px.imshow(
    heatmap_data,
    labels=dict(x="Year", y="Town", color="% Agent-Involved"),
    x=heatmap_data.columns,
    y=heatmap_data.index,
    color_continuous_scale='Blues',
    aspect="auto",
    title="Agent Involvement Rate by Town and Year (% Agent-Involved)"
)
fig.update_layout(yaxis={'categoryorder':'total ascending'})
fig.show()

# Seaborn static heatmap (optional, also saves as PNG)
plt.figure(figsize=(12, max(8, len(towns)*0.4)))
sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='Blues', cbar_kws={'label': '% Agent-Involved'})
plt.title('Agent Involvement Rate by Town and Year (% Agent-Involved)')
plt.ylabel('Town')
plt.xlabel('Year')
plt.tight_layout()
plt.savefig('agent_involved_by_town_heatmap.png')
plt.show() 