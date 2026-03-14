 # ---------------------------- dqn_inference.py ----------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

NUM_VMS = 10
GPU_TYPES = ['CPU-only', 'T4', 'V100', 'A100', 'V100M16', 'V100M32', 'P100', 'G2', 'G3']
gpu_index = {gpu: i for i, gpu in enumerate(GPU_TYPES)}
STATE_DIM = 3 + NUM_VMS + len(GPU_TYPES)  # exec_time, threshold, gpu_req, vm_loads, one-hot gpu

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

def gpu_spec_to_onehot(gpu_spec):
    vec = [0] * len(GPU_TYPES)
    key = gpu_spec if gpu_spec in gpu_index else 'CPU-only'
    vec[gpu_index[key]] = 1
    return vec

# Load model
model = DQN(STATE_DIM, NUM_VMS)
try:
    model.load_state_dict(torch.load("dqn_model.pth", map_location=torch.device('cpu')))
    model.eval()
except FileNotFoundError:
    print("[Warning] Trained model file 'dqn_model.pth' not found. Using untrained weights.")

def predict_vm(features, vm_loads=None, return_extended=False):
    try:
        exec_time = features[9] - features[7]
        exec_time = max(exec_time, 0.01)
    except:
        exec_time = 1.0

    required_gpus = features[4]
    gpu_spec = required_gpus[0] if required_gpus else 'CPU-only'

    if vm_loads is None:
        vm_loads = [0.1] * NUM_VMS
    else:
        vm_loads = [min(1.0, l) for l in vm_loads]

    state = np.array([
        exec_time,
        15.0,  # success threshold
        1.0 if gpu_spec != 'CPU-only' else 0.0,
        *vm_loads,
        *gpu_spec_to_onehot(gpu_spec)
    ], dtype=np.float32)

    with torch.no_grad():
        output = model(torch.tensor(state).unsqueeze(0))
        q_values = output.numpy().squeeze()

    selected_vm = int(np.argmax(q_values))
    confidence = float(q_values[selected_vm])

    result = {
        "selected_vm_id": selected_vm,
        "confidence": round(confidence, 4),
        "q_values": [round(float(q), 4) for q in q_values]
    }

    return result if return_extended else selected_vm