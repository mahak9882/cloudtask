import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import random
from collections import deque

# Hyperparameters
EPISODES = 5
GAMMA = 0.9
LR = 1e-3
BATCH_SIZE = 64

# Load and preprocess dataset
df = pd.read_csv("Dataset_1.csv", low_memory=False)

# Encode gpu_spec as integers
df['gpu_spec'] = pd.factorize(df['gpu_spec'].astype(str))[0]

# Convert necessary fields to float
df['cpu_milli'] = pd.to_numeric(df['cpu_milli'], errors='coerce')
df['gpu_milli'] = pd.to_numeric(df['gpu_milli'], errors='coerce')
df['num_gpu'] = pd.to_numeric(df['num_gpu'], errors='coerce')
df = df.dropna()

# Label reward: 1 if scheduled_time > 0, else 0 (not used in training but kept for reference)
df['success'] = df['scheduled_time'].apply(lambda x: 1 if x > 0 else 0)

# Reduce dataset size for faster testing
df = df.sample(n=10000, random_state=42)

# Define available VMs
vms = [
    {"cpu": 4000, "gpu": 1},
    {"cpu": 8000, "gpu": 2},
    {"cpu": 2000, "gpu": 1}
]

STATE_SIZE = 4
ACTION_SIZE = len(vms)

# Define DQN model
class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_SIZE, 64),
            nn.ReLU(),
            nn.Linear(64, ACTION_SIZE)
        )

    def forward(self, x):
        return self.net(x)

# Initialize
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DQN().to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()
memory = deque(maxlen=10000)

# Function to get state vector
def get_state(task):
    return torch.tensor([
        float(task['cpu_milli']),
        float(task['num_gpu']),
        float(task['gpu_milli']),
        float(task['gpu_spec'])
    ], dtype=torch.float32)

# Reward based on if VM can run the task
def get_reward(task, vm):
    return 1 if task['cpu_milli'] <= vm['cpu'] and task['num_gpu'] <= vm['gpu'] else -1

# Training loop
for ep in range(EPISODES):
    total_reward = 0

    for _, task in df.iterrows():
        action = random.randint(0, ACTION_SIZE - 1)
        vm = vms[action]
        state = get_state(task).to(device)
        reward = get_reward(task, vm)
        total_reward += reward
        next_state = state  # No environment dynamics

        memory.append((state, action, reward, next_state))

        if len(memory) >= BATCH_SIZE:
            batch = random.sample(memory, BATCH_SIZE)
            states, actions, rewards, next_states = zip(*batch)
            states = torch.stack(states).to(device)
            actions = torch.tensor(actions).to(device)
            rewards = torch.tensor(rewards).float().to(device)
            next_states = torch.stack(next_states).to(device)

            q_vals = model(states)
            q_next = model(next_states).detach()
            q_targets = q_vals.clone()

            for i in range(BATCH_SIZE):
                q_targets[i, actions[i]] = rewards[i] + GAMMA * torch.max(q_next[i])

            loss = loss_fn(q_vals, q_targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    if ep % 10 == 0:
        print(f"Episode {ep} | Total Reward: {total_reward} | Memory: {len(memory)}")

# Save model
torch.save(model.state_dict(), "dqn_vm_model.pth")
print("✅ DQN Training Completed and Model Saved as dqn_vm_model.pth")
