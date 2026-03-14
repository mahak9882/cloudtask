import torch
import pandas as pd
from dqn_train_vm import DQN, get_state

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_csv("Dataset_1.csv")
vms = [
    {"cpu": 4000, "gpu": 1},
    {"cpu": 8000, "gpu": 2},
    {"cpu": 2000, "gpu": 1}
]

model = DQN().to(device)
model.load_state_dict(torch.load("dqn_vm_model.pth"))
model.eval()

success = 0
for _, task in df.iterrows():
    scores = []
    for vm in vms:
        state = get_state(task, vm).to(device)
        with torch.no_grad():
            q_vals = model(state)
        scores.append(q_vals)

    best_action = torch.argmax(torch.stack(scores)).item()
    best_vm = vms[best_action]
    if task['cpu_milli'] <= best_vm['cpu'] and task['num_gpu'] <= best_vm['gpu']:
        success += 1

print("✅ Total successfully scheduled tasks (DQN):", success)
