import torch
import pandas as pd
import joblib
from dqn_train_vm_balanced import DQN, get_state, vms  # import same vms + function

df = pd.read_csv("Dataset_1.csv", low_memory=False)
df['gpu_spec'] = pd.factorize(df['gpu_spec'].astype(str))[0]
df['cpu_milli'] = pd.to_numeric(df['cpu_milli'], errors='coerce')
df['gpu_milli'] = pd.to_numeric(df['gpu_milli'], errors='coerce')
df['num_gpu'] = pd.to_numeric(df['num_gpu'], errors='coerce')
df = df.dropna()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DQN().to(device)
model.load_state_dict(torch.load("dqn_model_balanced_1.pth"))
model.eval()

# Track assignments
vm_task_counts = {vm['id']: 0 for vm in vms}
successful_tasks = 0

for _, task in df.iterrows():
    best_score = -float('inf')
    best_vm = None

    for vm in vms:
        state = get_state(task, vm).unsqueeze(0).to(device)
        with torch.no_grad():
            score = model(state).item()
        if score > best_score:
            best_score = score
            best_vm = vm

    if task['cpu_milli'] <= best_vm['cpu'] and task['num_gpu'] <= best_vm['gpu']:
        successful_tasks += 1
        vm_task_counts[best_vm['id']] += 1

# Output results
for vm_id in sorted(vm_task_counts.keys()):
    print(f"VM {vm_id} was assigned {vm_task_counts[vm_id]} tasks")

print(f"✅ Total successful tasks: {successful_tasks}/{len(df)} "
      f"({successful_tasks/len(df)*100:.2f}%)")

# Optional: save to CSV
pd.DataFrame([vm_task_counts]).to_csv("dqn_simulation_results_balanced.csv", index=False)
