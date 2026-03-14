 # ---------------------------- A2C_TRAIN.py ----------------------------
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import random

# --- Hyperparameters ---
NUM_VMS = 10
GAMMA = 0.99
LR = 1e-3
EPISODES = 30

# --- Environment Constants ---
GPU_TYPES = ['CPU-only', 'T4', 'V100', 'A100', 'V100M16', 'V100M32', 'P100', 'G2', 'G3']
gpu_index = {gpu: i for i, gpu in enumerate(GPU_TYPES)}
STATE_DIM = 6 + len(GPU_TYPES)

# --- Actor and Critic Networks ---
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.out(x), dim=-1)

class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)

# --- Helper Functions ---
def gpu_spec_to_onehot(gpu_spec):
    vec = [0] * len(GPU_TYPES)
    key = gpu_spec if gpu_spec in gpu_index else 'CPU-only'
    vec[gpu_index[key]] = 1
    return vec

def load_data(csv_path):
    df = pd.read_csv(csv_path, low_memory=False)
    df.fillna(0, inplace=True)

    qos_map = {'BestEffort': 0, 'Burstable': 1, 'Guaranteed': 2}
    phase_map = {'Running': 0, 'Pending': 1, 'Succeeded': 2, 'Failed': 3}
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
            15.0,  # updated success threshold
            0.1, 0.1, 0.1,  # dummy load values
            1.0 if gpu != 'CPU-only' else 0.0,
            *gpu_spec_to_onehot(gpu)
        ], dtype=np.float32)

        label = random.randint(0, NUM_VMS - 1)
        data.append((state, label))

    return data

# --- Training Function ---
def train_a2c(csv_file):
    data = load_data(csv_file)
    actor = Actor(STATE_DIM, NUM_VMS)
    critic = Critic(STATE_DIM)

    optimizerA = optim.Adam(actor.parameters(), lr=LR)
    optimizerC = optim.Adam(critic.parameters(), lr=LR)

    for ep in range(EPISODES):
        total_reward = 0
        random.shuffle(data)
        for state, action in data:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            dist = actor(state_tensor)
            value = critic(state_tensor)

            dist_np = dist.squeeze().detach().numpy()
            action_taken = action
            reward = 1.0  # optimistic reward

            next_value = value.detach()
            advantage = reward + GAMMA * next_value - value

            log_prob = torch.log(dist[0][action_taken] + 1e-8)
            actor_loss = -log_prob * advantage.detach()
            critic_loss = advantage.pow(2)

            optimizerA.zero_grad()
            actor_loss.backward()
            optimizerA.step()

            optimizerC.zero_grad()
            critic_loss.backward()
            optimizerC.step()

            total_reward += reward

        print(f"Episode {ep + 1}/{EPISODES} | Total Reward: {total_reward:.2f}")

    torch.save(actor.state_dict(), "a2c_actor.pth")
    torch.save(critic.state_dict(), "a2c_critic.pth")
    print("✅ A2C Model trained and saved.")

if __name__ == "__main__":
    train_a2c("merged_without_nan_scheduled_time.csv")
