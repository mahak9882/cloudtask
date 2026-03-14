# dqn_MODEL_grid.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
import random
from collections import deque
from sklearn.preprocessing import StandardScaler

CSV_PATH = "merged_without_nan_scheduled_time.csv"
NUM_VMS = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

def get_gpu_spec_list(df):
    specs = set()
    for s in df["gpu_spec"].dropna():
        for g in str(s).split("|"):
            specs.add(g.strip())
    return sorted(list(specs))

def load_data(csv_path, gpu_spec_list):
    qos_map = {'BestEffort': 0, 'Burstable': 1, 'Guaranteed': 2}
    phase_map = {'Running': 0, 'Pending': 1, 'Succeeded': 2, 'Failed': 3}
    df = pd.read_csv(csv_path, dtype=str)
    data = []

    for _, row in df.iterrows():
        gpu_onehot = [0] * len(gpu_spec_list)
        raw_spec = str(row.get('gpu_spec', '')).strip()
        if raw_spec:
            for g in raw_spec.split("|"):
                if g.strip() in gpu_spec_list:
                    gpu_onehot[gpu_spec_list.index(g.strip())] = 1

        def safe_float(x): return float(x.strip()) if str(x).strip() else 0.0

        features = [
            safe_float(row['cpu_milli']) / 1000.0,
            safe_float(row['memory_mib']) / 1024.0,
            safe_float(row['num_gpu']),
            safe_float(row['gpu_milli']) / 1000.0,
            qos_map.get(row['qos'], 0),
            phase_map.get(row['pod_phase'], 0),
            safe_float(row['creation_time']) / 1e6,
            safe_float(row['deletion_time']) / 1e6,
            safe_float(row['scheduled_time']) / 1e6
        ] + gpu_onehot

        exec_time = features[8] - features[6]
        exec_time = max(exec_time, 0.01)
        reward = max(0, 1.0 - (exec_time / 12.0))  # reward between 0 and 1
        label = random.randint(0, NUM_VMS - 1)
        data.append((features, label, reward))

    return data

def train_dqn(csv_path):
    df = pd.read_csv(csv_path, dtype=str)
    gpu_spec_list = get_gpu_spec_list(df)
    full_data = load_data(csv_path, gpu_spec_list)

    random.shuffle(full_data)
    full_data = full_data[:10000]  # Increase this if needed

    X = [row[0] for row in full_data]
    y = [row[1] for row in full_data]
    rewards = [row[2] for row in full_data]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    input_dim = len(X[0])
    model = DQN(input_dim, NUM_VMS).to(device)
    target_model = DQN(input_dim, NUM_VMS).to(device)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    memory = ReplayBuffer(10000)

    EPISODES = 500
    BATCH_SIZE = 64
    GAMMA = 0.95
    TARGET_UPDATE = 5

    best_reward_sum = -float('inf')

    for ep in range(EPISODES):
        combined = list(zip(X_scaled, y, rewards))
        random.shuffle(combined)
        total_loss = 0
        reward_sum = 0

        for features, action, reward in combined:
            state = torch.tensor(features, dtype=torch.float32).to(device)
            next_state = state.clone()
            memory.push((state, action, reward, next_state))
            if len(memory) < BATCH_SIZE:
                continue

            transitions = memory.sample(BATCH_SIZE)
            batch_state, batch_action, batch_reward, batch_next = zip(*transitions)

            batch_state = torch.stack(batch_state)
            batch_action = torch.tensor(batch_action, dtype=torch.long)
            batch_reward = torch.tensor(batch_reward, dtype=torch.float32)
            batch_next = torch.stack(batch_next)

            q_values = model(batch_state)
            next_q_values = target_model(batch_next)

            q_value = q_values.gather(1, batch_action.view(-1, 1)).squeeze(1)
            max_next_q = next_q_values.max(1)[0]
            expected_q = batch_reward + GAMMA * max_next_q

            loss = F.mse_loss(q_value, expected_q.detach())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            reward_sum += sum(batch_reward.tolist())

        if reward_sum > best_reward_sum:
            best_reward_sum = reward_sum
            torch.save(model.state_dict(), "dqn_model_trained_test.pth")
            print(f"[✓] New best model saved (Ep {ep+1})")

        if (ep + 1) % TARGET_UPDATE == 0:
            target_model.load_state_dict(model.state_dict())

        print(f"Episode {ep+1}/{EPISODES} - Loss: {total_loss:.4f} - Reward Sum: {reward_sum:.2f}")

    print("✅ Training complete.")

if __name__ == "__main__":
    train_dqn(CSV_PATH)

