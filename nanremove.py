import pandas as pd

# Load the CSV
df = pd.read_csv("dataset/merged_openb_all_with_source.csv")

# Drop rows where 'scheduled_time' is NaN
df_cleaned = df.dropna(subset=['scheduled_time'])

# Save to a new CSV file
df_cleaned.to_csv("dataset/merged_without_nan_scheduled_time.csv", index=False)

print("✅ New file saved as 'merged_without_nan_scheduled_time.csv' with NaNs in 'scheduled_time' removed.")
