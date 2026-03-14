# ---------------------------- TRAIN_DQN.py ----------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import random
from collections import deque

# Hyperparameters
EPISODES = 5
GAMMA = 0.95
LR = 1e-3
BATCH_SIZE = 64
MEMORY_SIZE = 10000
TARGET_UPDATE = 5
NUM_VMS = 10

# GPU Types
GPU_TYPES = ['CPU-only', 'T4', 'V100', 'A100', 'V100M16', 'V100M32', 'P100', 'G2', 'G3']
gpu_index = {gpu: i for i, gpu in enumerate(GPU_TYPES)}
STATE_DIM = 3 + NUM_VMS + len(GPU_TYPES)

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

def load_data(csv_path):
    df = pd.read_csv(csv_path, low_memory=False)
    df.fillna(0, inplace=True)
    data = []
    for _, row in df.iterrows():
        try:
            exec_time = float(row['deletion_time']) - float(row['creation_time'])
            exec_time = max(exec_time, 0.01)
        except:
            exec_time = 1.0
        gpu_spec = str(row.get('gpu_spec', '')).strip()
        gpu = gpu_spec if gpu_spec in gpu_index else 'CPU-only'
        state = np.array([
            exec_time, 15.0,
            1.0 if gpu != 'CPU-only' else 0.0,
            *[0.1]*NUM_VMS, *gpu_spec_to_onehot(gpu)
        ], dtype=np.float32)
        label = random.randint(0, NUM_VMS - 1)
        data.append((state, label))
    return data

def train_dqn(csv_file):
    data = load_data(csv_file)
    input_dim = len(data[0][0])
    model = DQN(input_dim, NUM_VMS)
    target_model = DQN(input_dim, NUM_VMS)
    target_model.load_state_dict(model.state_dict())
    optimizer = optim.Adam(model.parameters(), lr=LR)
    memory = deque(maxlen=MEMORY_SIZE)

    for ep in range(EPISODES):
        total_loss = 0
        random.shuffle(data)
        for state, action in data:
            state_tensor = torch.tensor(state, dtype=torch.float32)
            reward = 1.0
            memory.append((state_tensor, action, reward, state_tensor))

            if len(memory) < BATCH_SIZE:
                continue
            batch = random.sample(memory, BATCH_SIZE)
            states, actions, rewards, next_states = zip(*batch)

            states = torch.stack(states)
            next_states = torch.stack(next_states)
            actions = torch.tensor(actions)
            rewards = torch.tensor(rewards)

            q_vals = model(states).gather(1, actions.unsqueeze(1)).squeeze()
            max_next_q_vals = target_model(next_states).max(1)[0]
            expected_q = rewards + GAMMA * max_next_q_vals
            loss = F.mse_loss(q_vals, expected_q.detach())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (ep + 1) % TARGET_UPDATE == 0:
            target_model.load_state_dict(model.state_dict())

        print(f"Episode {ep+1}/{EPISODES} - Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "dqn_model_1.pth")
    print("✅ Model saved as dqn_model_1.pth")

def predict_vm(state):
    model = DQN(len(state), NUM_VMS)
    model.load_state_dict(torch.load("dqn_model_1.pth", map_location=torch.device('cpu')))
    model.eval()
    with torch.no_grad():
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        q_values = model(state_tensor)
        vm = torch.argmax(q_values).item()
        confidence = torch.softmax(q_values, dim=1)[0][vm].item()
    return vm, confidence

if __name__ == "__main__":
    train_dqn("Dataset_1.csv")
