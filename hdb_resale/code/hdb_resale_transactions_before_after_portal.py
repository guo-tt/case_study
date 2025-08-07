import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# Path to the HDB resale flat prices CSV (update this path as needed)
CSV_PATH = '../data/hdb_resale_flat_prices.csv'  # Adjust if your data is elsewhere

# Load the data
df = pd.read_csv(CSV_PATH)

# Ensure the 'month' column is datetime
df['month'] = pd.to_datetime(df['month'], format='%Y-%m')
df['year'] = df['month'].dt.year

# Filter for years 2017-2024
df = df[(df['year'] >= 2017) & (df['year'] <= 2024)]

# Group by year and count transactions
transactions_per_year = df.groupby('year').size().reset_index(name='total_transactions')

# Print the table
print('HDB Resale Transactions: Before and After the Portal (2017-2024)')
print(transactions_per_year.to_string(index=False))

# Seaborn static plot
plt.figure(figsize=(8,5))
sns.lineplot(data=transactions_per_year, x='year', y='total_transactions', marker='o', color='#1976d2')
plt.title('Total HDB Resale Transactions (2017–2024)')
plt.xlabel('Year')
plt.ylabel('Number of Transactions')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# Plotly interactive plot
fig = px.line(transactions_per_year, x='year', y='total_transactions', markers=True,
              title='Total HDB Resale Transactions (2017–2024)',
              labels={'year': 'Year', 'total_transactions': 'Number of Transactions'})
fig.update_traces(line_color='#1976d2')
fig.show()

# Optionally, save the table and static chart
transactions_per_year.to_csv('hdb_resale_transactions_2017_2024.csv', index=False)
plt.savefig('hdb_resale_transactions_2017_2024.png') 