import pandas as pd
import plotly.express as px

# Paths to the datasets (update as needed)
CEA_CSV = '../data/CEA_SalespersonsPropertyTransactionRecordsresidential.csv'  # CEA agent transactions
HDB_CSV = '../data/hdb_resale_flat_prices.csv'  # All HDB resale transactions

# Load datasets
df_cea = pd.read_csv(CEA_CSV)
df_hdb = pd.read_csv(HDB_CSV)

# --- Preprocess CEA data ---
# Ensure transaction_date is datetime (e.g., 'JAN-2018')
df_cea = df_cea[df_cea['property_type'].str.upper() == 'HDB']
df_cea = df_cea[df_cea['transaction_type'].str.upper() == 'RESALE']
df_cea = df_cea[df_cea['transaction_date'].notnull()]
df_cea['year'] = pd.to_datetime(df_cea['transaction_date'], format='%b-%Y', errors='coerce').dt.year

# Only keep valid years
cea_years = df_cea['year'].dropna().astype(int)
df_cea = df_cea.loc[cea_years.index]
df_cea['year'] = cea_years.values

# --- Preprocess HDB data ---
df_hdb['month'] = pd.to_datetime(df_hdb['month'], format='%Y-%m')
df_hdb['year'] = df_hdb['month'].dt.year

# Filter for years 2017-2024
years = list(range(2017, 2025))

# Count agent-involved transactions per year
agent_involved = df_cea[df_cea['year'].isin(years)].groupby('year').size().reindex(years, fill_value=0)

# Count total HDB resale transactions per year
total_transactions = df_hdb[df_hdb['year'].isin(years)].groupby('year').size().reindex(years, fill_value=0)

direct_no_agent = total_transactions - agent_involved
percent_agent_involved = (agent_involved / total_transactions * 100).round(2).replace([float('inf'), float('nan')], 0)

# Create result table
result = pd.DataFrame({
    'Year': years,
    'Total HDB Resale Transactions': total_transactions.values,
    'Agent-Involved': agent_involved.values,
    'Direct (No Agent)': direct_no_agent.values,
    '% Agent-Involved': percent_agent_involved.values
})

print('Agent-Involved vs Direct HDB Resale Transactions (2017-2024)')
print(result.to_string(index=False))

# Optionally, save the table
result.to_csv('agent_vs_direct_hdb_resale_2017_2024.csv', index=False)

# Interactive bar chart for Agent-Involved
fig = px.bar(result, x='Year', y='Agent-Involved',
             title='Agent-Involved HDB Resale Transactions per Year (2017–2024)',
             labels={'Agent-Involved': 'Agent-Involved Transactions'},
             text='Agent-Involved', color='Agent-Involved', color_continuous_scale='Blues')
fig.update_traces(texttemplate='%{text}', textposition='outside')
fig.update_layout(yaxis_title='Number of Agent-Involved Transactions', xaxis_title='Year')
fig.show()

# Interactive line plot for % Agent-Involved ratio
fig2 = px.line(result, x='Year', y='% Agent-Involved', markers=True,
               title='Agent-Involved Ratio in HDB Resale Transactions per Year (2017–2024)',
               labels={'% Agent-Involved': 'Agent-Involved Ratio (%)'})
fig2.update_traces(line_color='#388e3c', marker=dict(size=10))
fig2.update_layout(yaxis_title='Agent-Involved Ratio (%)', xaxis_title='Year', yaxis_range=[0, 100])
fig2.show() 