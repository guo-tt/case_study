"""
Configuration file for Preschool Demand Forecasting System
"""

# System Configuration
SYSTEM_CONFIG = {
    'max_children_per_center': 100,
    'enrollment_rate': 0.8,
    'forecast_horizon_years': 5,
    'bto_occupancy_rate': 0.15,  # 15% of BTO residents are children 18m-6y
    'persons_per_bto_unit': 2.5,
    'capacity_threshold': 50  # Minimum gap to trigger recommendation
}

# Data Sources
DATA_SOURCES = {
    'population_data': '../data/respopagesex2000to2020e.xlsx',
    'bto_mapping': '../data/btomapping.csv',
    'preschool_centers': '../data/ListingofCentres.csv'
}

# Geographic Configuration
REGIONS = {
    'Central Region': ['Bishan', 'Bukit Merah', 'Bukit Timah', 'Geylang', 'Kallang', 'Marine Parade', 'Novena', 'Queenstown', 'Southern Islands', 'Tanglin', 'Toa Payoh'],
    'East Region': ['Bedok', 'Changi', 'Pasir Ris', 'Paya Lebar', 'Tampines'],
    'North Region': ['Central Water Catchment', 'Lim Chu Kang', 'Mandai', 'Sembawang', 'Simpang', 'Sungei Kadut', 'Woodlands', 'Yishun'],
    'North-East Region': ['Ang Mo Kio', 'Hougang', 'North-Eastern Islands', 'Punggol', 'Seletar', 'Sengkang', 'Serangoon'],
    'West Region': ['Boon Lay', 'Bukit Batok', 'Bukit Panjang', 'Choa Chu Kang', 'Clementi', 'Jurong East', 'Jurong West', 'Pioneer', 'Tengah', 'Tuas', 'Western Islands', 'Western Water Catchment']
}

# Age Groups for Preschool
AGE_GROUPS = {
    'infant': '18 months - 2 years',
    'pg': '2-3 years (Playgroup)',
    'n1': '3-4 years (Nursery 1)',
    'n2': '4-5 years (Nursery 2)',
    'k1': '5-6 years (Kindergarten 1)',
    'k2': '6-7 years (Kindergarten 2)'
}

# Service Models
SERVICE_MODELS = {
    'CC': 'Child Care',
    'KN': 'Kindergarten',
    'DS': 'Development Support',
    'EYC-DS': 'Early Years Centre - Development Support'
}

# Output Configuration
OUTPUT_CONFIG = {
    'results_file': 'preschool_forecast_results.csv',
    'dashboard_file': 'preschool_forecast_dashboard.png',
    'recommendations_file': 'preschool_recommendations.csv',
    'report_file': 'preschool_forecast_report.html'
} 