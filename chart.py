import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# --- 1. Seaborn Best Practices: Set Professional Style and Context ---
# Use 'whitegrid' for a clean, professional background
sns.set_style("whitegrid")
# Use 'poster' or 'talk' context for presentation-ready text sizes
# 'talk' is a good balance for executive reports
sns.set_context("talk") 

# --- 2. Data Generation: Create realistic synthetic data ---
# Define the customer engagement metrics
metrics = [
    'Monthly_Visits',
    'Avg_Time_Spent_min',
    'Purchase_Frequency',
    'AOV_USD', # Average Order Value
    'CSAT_Score', # Customer Satisfaction Score
    'Support_Tickets'
]

# Generate synthetic data with realistic interdependencies
# We use a multivariate normal distribution to ensure realistic correlations
# Define a desired correlation matrix (realistic expectations for retail)
# High positive correlation: Visits & Time, Visits & Purchase Frequency
# Moderate positive correlation: Purchase Freq & AOV, CSAT & Purchase Freq
# Negative correlation: Support Tickets with CSAT and Purchase Freq
desired_corr = np.array([
    [1.0, 0.70, 0.65, 0.40, 0.20, -0.30],
    [0.70, 1.0, 0.55, 0.35, 0.15, -0.25],
    [0.65, 0.55, 1.0, 0.50, 0.45, -0.55],
    [0.40, 0.35, 0.50, 1.0, 0.30, -0.10],
    [0.20, 0.15, 0.45, 0.30, 1.0, -0.60],
    [-0.30, -0.25, -0.55, -0.10, -0.60, 1.0]
])

# Use the desired correlation matrix to generate a covariance matrix
# Assuming standard deviations (std devs) for data columns
stds = np.array([4, 10, 1.5, 50, 0.8, 2])
cov = np.outer(stds, stds) * desired_corr

# Generate 500 samples of correlated data
np.random.seed(42) # for reproducibility
data = np.random.multivariate_normal(
    mean=[10, 30, 3, 150, 4.5, 4], # Realistic means
    cov=cov,
    size=500
)

# Convert to DataFrame
df = pd.DataFrame(data, columns=metrics)

# Clip values to ensure they are realistic (e.g., no negative counts)
df['Monthly_Visits'] = df['Monthly_Visits'].clip(lower=0).round().astype(int)
df['Purchase_Frequency'] = df['Purchase_Frequency'].clip(lower=0.1)
df['AOV_USD'] = df['AOV_USD'].clip(lower=20)
df['CSAT_Score'] = df['CSAT_Score'].clip(lower=1, upper=5).round(2)
df['Support_Tickets'] = df['Support_Tickets'].clip(lower=0).round().astype(int)

# Calculate the final correlation matrix for the visualization
corr_matrix = df.corr()

# --- 3. Create Heatmap Visualization ---

# Set figure size for the required 512x512 output
# To get 512x512 pixels with dpi=64, we need a figure size of 512/64 = 8
plt.figure(figsize=(8, 8)) 

# Create the heatmap
sns.heatmap(
    corr_matrix, 
    annot=True,              # Show the correlation values on the heatmap
    fmt=".2f",               # Format the annotation to two decimal places
    cmap="vlag",             # Professional divergent color palette 
    linewidths=.5,           # Add lines to separate cells
    linecolor='black',       # Line color for better separation
    cbar_kws={'label': 'Correlation Coefficient'} # Label for the color bar
)

# --- 4. Style the Chart: Titles and Labels ---

# Set a professional title
plt.title(
    "Customer Engagement Correlation Matrix Heatmap",
    fontsize=18,
    fontweight='bold',
    pad=20
)

# Improve axis labels for better readability (rotation)
plt.yticks(rotation=0)
plt.xticks(rotation=45, ha='right')

# Adjust layout to prevent labels from being cut off
plt.tight_layout()

# --- 5. Export: Save as PNG with required dimensions ---
# 512x512 output requires (8, 8) figsize and dpi=64: 8 inches * 64 dpi = 512 pixels
plt.savefig(
    'chart.png', 
    dpi=64, 
    bbox_inches='tight' # Ensures no borders/labels are cut off
)

print("Heatmap successfully generated and saved as 'chart.png' with 512x512 dimensions.")
print("Generated Correlation Matrix:")
print(corr_matrix.round(2))
