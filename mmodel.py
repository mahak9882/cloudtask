import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import json

# ------------------- 1. Globals -------------------
GPU_TYPES = ['CPU-only', 'T4', 'V100', 'A100', 'V100M16', 'V100M32', 'P100', 'G2', 'G3']
NUM_VMS = 10
SUCCESS_RATE_THRESHOLD = 12.0
STATE_DIM = 6 + len(GPU_TYPES)

gpu_index = {gpu: i for i, gpu in enumerate(GPU_TYPES)}

# ------------------- 2. DQN Model -------------------
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

model = DQN(STATE_DIM, NUM_VMS)
try:
    model.load_state_dict(torch.load("dataset/dqn_model.pth", map_location=torch.device('cpu')))
    model.eval()
except FileNotFoundError:
    print("[Warning] Trained model file 'dqn_model.pth' not found. Using untrained weights.")

# ------------------- 3. GPU One-Hot -------------------
def gpu_spec_to_onehot(gpu_spec):
    vec = [0] * len(GPU_TYPES)
    key = gpu_spec if gpu_spec in gpu_index else 'CPU-only'
    vec[gpu_index[key]] = 1
    return vec

# ------------------- 4. Inference -------------------
def select_vm(task_info, vm_loads):
    gpu_vec = gpu_spec_to_onehot(task_info['gpu_spec'])

    state = np.array([
        task_info['exec_time'],
        SUCCESS_RATE_THRESHOLD,
        *vm_loads[:3],
        task_info['gpu_req'],
        *gpu_vec
    ], dtype=np.float32)

    state_tensor = torch.tensor(state).unsqueeze(0)
    with torch.no_grad():
        q_values = model(state_tensor).squeeze().numpy()

    selected_vm = int(np.argmax(q_values))
    confidence = float(q_values[selected_vm])
    return {
        "selected_vm_id": selected_vm,
        "confidence": round(confidence, 4),
        "q_values": [round(float(q), 4) for q in q_values]
    }

# ------------------- 5. For SimPy -------------------
global_vm_list = []

def predict_vm(features, return_extended=False):
    try:
        exec_time = features[9] - features[7]
        exec_time = max(exec_time, 0.01)
    except:
        exec_time = 1.0

    required_gpus = features[4]
    gpu_spec = required_gpus[0] if required_gpus else 'CPU-only'

    task_info = {
        "exec_time": exec_time,
        "gpu_req": 1.0 if gpu_spec != 'CPU-only' else 0.0,
        "gpu_spec": gpu_spec
    }

    vm_loads = [0.1] * NUM_VMS
    result = select_vm(task_info, vm_loads)

    return result if return_extended else result["selected_vm_id"]

# ------------------- 6. Batch Inference -------------------
def run_batch_inference(csv_path, output_file=None):
    df = pd.read_csv(csv_path)
    df.fillna(0, inplace=True)

    results = []

    for i, row in df.iterrows():
        try:
            exec_time = float(row.get('deletion_time', 0)) - float(row.get('creation_time', 0))
            exec_time = max(exec_time, 0.01)
        except:
            exec_time = 1.0

        gpu_spec_raw = str(row.get("gpu_spec", '')).strip()
        gpu_spec = gpu_spec_raw if gpu_spec_raw in GPU_TYPES else 'CPU-only'

        task_info = {
            "exec_time": exec_time,
            "gpu_req": 1.0 if gpu_spec != 'CPU-only' else 0.0,
            "gpu_spec": gpu_spec
        }

        vm_loads = [0.1] * NUM_VMS
        result = select_vm(task_info, vm_loads)

        task_id = row.get("name", f"task_{i}")
        output = {task_id: result}
        print(json.dumps(output))

        results.append(output)

    if output_file:
        with open(output_file, "w") as f:
            for r in results:
                json.dump(r, f)
                f.write("\n")

    return results

# ------------------- 7. Run if needed -------------------
if __name__ == "__main__":
    run_batch_inference("merged_without_nan_scheduled_time.csv", output_file="vm_predictions.jsonl")
