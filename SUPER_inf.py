 # ---------------------------- supervised_inference.py ----------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

# Constants
NUM_VMS = 10
GPU_TYPES = ['CPU-only', 'T4', 'V100', 'A100', 'V100M16', 'V100M32', 'P100', 'G2', 'G3']
gpu_index = {gpu: i for i, gpu in enumerate(GPU_TYPES)}
STATE_DIM = 6 + len(GPU_TYPES)

# --- Model ---
class SupervisedVMClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SupervisedVMClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

# --- One-hot encoding ---
def gpu_spec_to_onehot(gpu_spec):
    vec = [0] * len(GPU_TYPES)
    key = gpu_spec if gpu_spec in gpu_index else 'CPU-only'
    vec[gpu_index[key]] = 1
    return vec

# --- Load trained model ---
model = SupervisedVMClassifier(STATE_DIM, NUM_VMS)
try:
    model.load_state_dict(torch.load("supervised_vm_model.pth", map_location=torch.device('cpu')))
    model.eval()
except FileNotFoundError:
    print("[Warning] Model file 'supervised_vm_model.pth' not found. Please train it first using SUPER_TRAIN.py")

# --- Predict VM ---
def predict_vm_supervised(features, return_extended=False):
    try:
        exec_time = features[9] - features[7]
        exec_time = max(exec_time, 0.01)
    except:
        exec_time = 1.0

    required_gpus = features[4]
    gpu_spec = required_gpus[0] if required_gpus else 'CPU-only'

    state = np.array([
        exec_time,
        15.0,
        0.1, 0.1, 0.1,
        1.0 if gpu_spec != 'CPU-only' else 0.0,
        *gpu_spec_to_onehot(gpu_spec)
    ], dtype=np.float32)

    with torch.no_grad():
        output = model(torch.tensor(state).unsqueeze(0))
        probs = F.softmax(output, dim=1).numpy().squeeze()

    selected_vm = int(np.argmax(probs))
    confidence = float(probs[selected_vm])

    result = {
        "selected_vm_id": selected_vm,
        "confidence": round(confidence, 4),
        "probs": [round(float(p), 4) for p in probs]
    }

    return result if return_extended else selected_vm
