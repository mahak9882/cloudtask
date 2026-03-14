# ---------------------------- TRAIN_DQN_1.py ----------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import random
from collections import deque

# Hyperparameters
EPISODES = 50
GAMMA = 0.95
LR = 1e-3
BATCH_SIZE = 64
MEMORY_SIZE = 10000
TARGET_UPDATE = 5
NUM_VMS = 10

# Environment Constants
GPU_TYPES = ['CPU-only', 'T4', 'V100', 'A100', 'V100M16', 'V100M32', 'P100', 'G2', 'G3']
gpu_index = {gpu: i for i, gpu in enumerate(GPU_TYPES)}
STATE_DIM = 3 + NUM_VMS + len(GPU_TYPES)  # exec_time, threshold, gpu_req, vm_loads, one-hot gpu

# DQN Model
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

# Experience Replay
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# GPU to One-hot
def gpu_spec_to_onehot(gpu_spec):
    vec = [0] * len(GPU_TYPES)
    key = gpu_spec if gpu_spec in gpu_index else 'CPU-only'
    vec[gpu_index[key]] = 1
    return vec

# Load Dataset
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
            exec_time,
            15.0,
            1.0 if gpu != 'CPU-only' else 0.0,
            *[0.1]*NUM_VMS,
            *gpu_spec_to_onehot(gpu)
        ], dtype=np.float32)

        label = random.randint(0, NUM_VMS - 1)
        data.append((state, label))

    return data

# Train Function
def train_dqn(csv_file):
    data = load_data(csv_file)
    input_dim = len(data[0][0])

    model = DQN(input_dim, NUM_VMS)
    target_model = DQN(input_dim, NUM_VMS)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()

    optimizer = optim.Adam(model.parameters(), lr=LR)
    memory = ReplayBuffer(MEMORY_SIZE)

    for ep in range(EPISODES):
        random.shuffle(data)
        total_loss = 0

        for state, action in data:
            state_tensor = torch.tensor(state, dtype=torch.float32)
            next_state_tensor = state_tensor.clone()

            reward = 1.0
            memory.push((state_tensor, action, reward, next_state_tensor))

            if len(memory) < BATCH_SIZE:
                continue

            transitions = memory.sample(BATCH_SIZE)
            states, actions, rewards, next_states = zip(*transitions)

            states = torch.tensor(np.array(states), dtype=torch.float32)
            next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
            actions = torch.tensor(actions, dtype=torch.long)
            rewards = torch.tensor(rewards, dtype=torch.float32)

            q_values = model(states)
            next_q_values = target_model(next_states)

            q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze()
            max_next_q_value = next_q_values.max(1)[0]
            expected_q = rewards + GAMMA * max_next_q_value

            loss = F.mse_loss(q_value, expected_q.detach())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (ep + 1) % TARGET_UPDATE == 0:
            target_model.load_state_dict(model.state_dict())

        print(f"Episode {ep+1}/{EPISODES} - Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "dqn_model.pth")
    print("✅ DQN model trained and saved as 'dqn_model.pth'")

# --- Run ---
if __name__ == "__main__":
    train_dqn("merged_without_nan_scheduled_time.csv")

