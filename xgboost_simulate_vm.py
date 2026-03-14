import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("Dataset_1.csv", low_memory=False)

# Encode gpu_spec same way as training
le = LabelEncoder()
df['gpu_spec'] = le.fit_transform(df['gpu_spec'].astype(str))

# Ensure all feature columns are numeric
df['cpu_milli'] = pd.to_numeric(df['cpu_milli'], errors='coerce')
df['num_gpu'] = pd.to_numeric(df['num_gpu'], errors='coerce')
df['gpu_milli'] = pd.to_numeric(df['gpu_milli'], errors='coerce')
df['gpu_spec'] = pd.to_numeric(df['gpu_spec'], errors='coerce')

# Drop rows with any NaNs (if any conversion failed)
df = df.dropna()

# Load model
model = joblib.load("xgboost_vm_model.pkl")

# VM configs
vms = [
    {"id": 0, "cpu": 4000, "gpu": 1},
    {"id": 1, "cpu": 8000, "gpu": 2},
    {"id": 2, "cpu": 2000, "gpu": 1}
]

success_count = 0
total_tasks = len(df)

for idx, row in df.iterrows():
    best_score = -1
    best_vm = None

    for vm in vms:
        # Convert input to DataFrame with exact column names
        features_df = pd.DataFrame([{
            'cpu_milli': row['cpu_milli'],
            'num_gpu': row['num_gpu'],
            'gpu_milli': row['gpu_milli'],
            'gpu_spec': row['gpu_spec']
        }])
        # Predict success probability
        score = model.predict_proba(features_df)[0][1]

        if score > best_score:
            best_score = score
            best_vm = vm

    # Simulate VM assignment
    if row['cpu_milli'] <= best_vm['cpu'] and row['num_gpu'] <= best_vm['gpu']:
        success_count += 1

print(f"✅ Successful tasks scheduled: {success_count}/{total_tasks} ({(success_count/total_tasks)*100:.2f}%)")

