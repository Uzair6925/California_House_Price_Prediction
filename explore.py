from sklearn.datasets import fetch_california_housing
import pandas as pd

# Load California Housing Dataset
data = fetch_california_housing()

# Convert dataset into DataFrame
df = pd.DataFrame(data.data, columns=data.feature_names)

# Add target column
df["Price"] = data.target

# Explore dataset
print("Shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nStatistical Summary:")
print(df.describe())

