import pandas as pd
import numpy as np
import plotly.express as px

# User can change this year
YEAR = 2023

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

df_cea = df_cea[df_cea['year'] == YEAR]

# --- Preprocess HDB data ---
df_hdb['month'] = pd.to_datetime(df_hdb['month'], format='%Y-%m')
df_hdb['year'] = df_hdb['month'].dt.year
df_hdb = df_hdb[df_hdb['year'] == YEAR]

# --- By Town ---
towns = sorted(df_hdb['town'].unique())
results_town = []
for town in towns:
    total = df_hdb[df_hdb['town'] == town].shape[0]
    agent = df_cea[df_cea['town'] == town].shape[0]
    pct_agent = (agent / total * 100) if total > 0 else np.nan
    results_town.append({'Town': town, '% Agent-Involved': round(pct_agent, 2), 'Total': total})
df_town = pd.DataFrame(results_town).sort_values('% Agent-Involved', ascending=False)

# Plotly bar chart by town
fig_town = px.bar(df_town, x='Town', y='% Agent-Involved',
                  title=f'Agent Involvement Rate by Town in {YEAR}',
                  labels={'% Agent-Involved': 'Agent-Involved Rate (%)'},
                  text='% Agent-Involved', color='% Agent-Involved', color_continuous_scale='Blues')
fig_town.update_traces(texttemplate='%{text}', textposition='outside')
fig_town.update_layout(yaxis_title='Agent-Involved Rate (%)', xaxis_title='Town', xaxis_tickangle=-45)
fig_town.show()
fig_town.write_image(f'agent_involved_rate_by_town_{YEAR}.png')

# --- By Flat Type ---
flat_types = sorted(df_hdb['flat_type'].unique())
results_type = []
for flat in flat_types:
    total = df_hdb[df_hdb['flat_type'] == flat].shape[0]
    agent = df_cea[df_cea['flat_type'] == flat].shape[0]
    pct_agent = (agent / total * 100) if total > 0 else np.nan
    results_type.append({'Flat Type': flat, '% Agent-Involved': round(pct_agent, 2), 'Total': total})
df_type = pd.DataFrame(results_type).sort_values('% Agent-Involved', ascending=False)

# Plotly bar chart by flat type
fig_type = px.bar(df_type, x='Flat Type', y='% Agent-Involved',
                  title=f'Agent Involvement Rate by Flat Type in {YEAR}',
                  labels={'% Agent-Involved': 'Agent-Involved Rate (%)'},
                  text='% Agent-Involved', color='% Agent-Involved', color_continuous_scale='Blues')
fig_type.update_traces(texttemplate='%{text}', textposition='outside')
fig_type.update_layout(yaxis_title='Agent-Involved Rate (%)', xaxis_title='Flat Type')
fig_type.show()