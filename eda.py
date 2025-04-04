import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
file_path = 'eeg_features_toy.csv'
df = pd.read_csv(file_path)

# Display first few rows
print(df.head())

# Dataset shape
num_samples, num_features = df.shape
print(f'Πλήθος Δειγμάτων: {num_samples}')
print(f'Πλήθος Χαρακτηριστικών: {num_features}')

# Feature types
print('Είδος Χαρακτηριστικών:')
print(df.dtypes)

# Check for missing values
missing_values = df.isnull().sum()
print("Missing values per column:")
print(missing_values[missing_values > 0])  # Show only columns with missing values

# Class distribution
samples_per_category = df['seizure'].value_counts()
print('Δείγματα ανά Κατηγορία (Seizure or not):')
print(samples_per_category)

# Drop non-numeric columns before computing correlation
numeric_df = df.select_dtypes(include=['number'])  # Select only numeric columns
numeric_df = numeric_df.drop(columns=['epoch', 'recording', 'second'])
correlation_with_seizure = numeric_df.corr()['seizure'].drop('seizure').sort_values(ascending=False)

print("Correlation of numeric features with 'Seizure':")
print(correlation_with_seizure)

# Convert to DataFrame for heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_with_seizure.to_frame(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Feature Correlation with Seizure")
plt.show()
