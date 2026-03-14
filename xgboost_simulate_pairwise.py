import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("Dataset_1.csv", low_memory=False)

# Encode and clean
df['gpu_spec'] = pd.factorize(df['gpu_spec'].astype(str))[0]
df['cpu_milli'] = pd.to_numeric(df['cpu_milli'], errors='coerce')
df['gpu_milli'] = pd.to_numeric(df['gpu_milli'], errors='coerce')
df['num_gpu'] = pd.to_numeric(df['num_gpu'], errors='coerce')
df = df.dropna()

model = joblib.load("xgboost_vm_selector.pkl")

vms = [
    {"id": 0, "cpu": 4000, "gpu": 1},
    {"id": 1, "cpu": 8000, "gpu": 2},
    {"id": 2, "cpu": 2000, "gpu": 1}
]

success_count = 0

for _, row in df.iterrows():
    best_score = -1
    best_vm = None

    for vm in vms:
        features = pd.DataFrame([{
            "cpu_milli": row['cpu_milli'],
            "num_gpu": row['num_gpu'],
            "gpu_milli": row['gpu_milli'],
            "gpu_spec": row['gpu_spec'],
            "vm_cpu": vm['cpu'],
            "vm_gpu": vm['gpu'],
        }])

        score = model.predict_proba(features)[0][1]

        if score > best_score:
            best_score = score
            best_vm = vm

    if row['cpu_milli'] <= best_vm['cpu'] and row['num_gpu'] <= best_vm['gpu']:
        success_count += 1

print(f"✅ Improved success rate: {success_count}/{len(df)} ({(success_count/len(df))*100:.2f}%)")
