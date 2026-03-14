# model1.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

GPU_TYPES = ['CPU-only', 'T4', 'V100', 'A100', 'V100M16', 'V100M32', 'P100', 'G2', 'G3']
NUM_VMS = 5
SUCCESS_RATE_THRESHOLD = 12.0
STATE_DIM = 3 + NUM_VMS + len(GPU_TYPES)  # exec_time, gpu_req, threshold, vm_loads, gpu_onehot

gpu_index = {gpu: i for i, gpu in enumerate(GPU_TYPES)}

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

model = DQN(STATE_DIM, NUM_VMS)
try:
    model.load_state_dict(torch.load("dqn_model_trained_new.pth", map_location=torch.device('cpu')))
    model.eval()
    print("Loaded trained DQN model.")
except FileNotFoundError:
    print("[Warning] Trained model not found. Using untrained weights.")

def gpu_spec_to_onehot(gpu_spec):
    vec = [0] * len(GPU_TYPES)
    if gpu_spec in gpu_index:
        vec[gpu_index[gpu_spec]] = 1
    return vec

global_vm_list = []

def select_vm(task_info, vm_loads):
    gpu_vec = gpu_spec_to_onehot(task_info['gpu_spec'])

    state = np.array([
        task_info['exec_time'],
        task_info['gpu_req'],
        SUCCESS_RATE_THRESHOLD,
        *vm_loads[:NUM_VMS],
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

    vm_loads = [len(vm.resource.queue) for vm in global_vm_list]
    result = select_vm(task_info, vm_loads)
    return result if return_extended else result["selected_vm_id"]
