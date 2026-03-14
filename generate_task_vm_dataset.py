import pandas as pd

# Load tasks
df = pd.read_csv("Dataset_1.csv", low_memory=False)

# Encode gpu_spec
df['gpu_spec'] = df['gpu_spec'].astype(str)
df['gpu_spec'] = pd.factorize(df['gpu_spec'])[0]

# Convert numeric columns
df['cpu_milli'] = pd.to_numeric(df['cpu_milli'], errors='coerce')
df['gpu_milli'] = pd.to_numeric(df['gpu_milli'], errors='coerce')
df['num_gpu'] = pd.to_numeric(df['num_gpu'], errors='coerce')
df = df.dropna()

# Define available VMs
vms = [
    {"id": 0, "cpu": 4000, "gpu": 1},
    {"id": 1, "cpu": 8000, "gpu": 2},
    {"id": 2, "cpu": 2000, "gpu": 1}
]

expanded_data = []

for _, row in df.iterrows():
    for vm in vms:
        label = 1 if row['scheduled_time'] > 0 and row['cpu_milli'] <= vm['cpu'] and row['num_gpu'] <= vm['gpu'] else 0

        expanded_data.append({
            "cpu_milli": row['cpu_milli'],
            "num_gpu": row['num_gpu'],
            "gpu_milli": row['gpu_milli'],
            "gpu_spec": row['gpu_spec'],
            "vm_cpu": vm['cpu'],
            "vm_gpu": vm['gpu'],
            "success": label
        })

# Save expanded dataset
df_expanded = pd.DataFrame(expanded_data)
df_expanded.to_csv("task_vm_training_data.csv", index=False)
print("✅ Expanded dataset saved as task_vm_training_data.csv")
