# grid_search_dqn.py
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import os

# --- Hyperparameter grid --- #
hyperparams_grid = [
    {'lr': 0.001, 'gamma': 0.9, 'epsilon': 0.1},
    {'lr': 0.0005, 'gamma': 0.95, 'epsilon': 0.2},
    {'lr': 0.0001, 'gamma': 0.99, 'epsilon': 0.3},
]

# --- DQN Model --- #
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_size)
        )

    def forward(self, x):
        return self.fc(x)

# --- Environment Functions --- #
def get_state(task):
    return torch.tensor([task['cpu_milli'], task['gpu_milli'], task['num_gpu'], task['gpu_spec']], dtype=torch.float32)

def get_reward(vm_resources, task):
    if task['cpu_milli'] <= vm_resources['cpu'] and task['num_gpu'] <= vm_resources['gpu']:
        return 1.0  # success
    return -5.0  # overload

def update_vm_resources(vm_resources, task):
    vm_resources['cpu'] -= task['cpu_milli']
    vm_resources['gpu'] -= task['num_gpu']

# --- Grid Search Training Loop --- #
results = []

df = pd.read_csv("Dataset_1.csv", low_memory=False)
df.dropna(subset=['cpu_milli', 'gpu_milli', 'num_gpu', 'gpu_spec'], inplace=True)
df['gpu_spec'] = pd.factorize(df['gpu_spec'].astype(str))[0]

vm_list = [{'cpu': 64000, 'gpu': 8} for _ in range(10)]

tasks = df[['cpu_milli', 'gpu_milli', 'num_gpu', 'gpu_spec']].to_dict(orient='records')

for idx, params in enumerate(hyperparams_grid):
    print(f"Training with {params}...")
    model = DQN(input_size=4, output_size=10)
    optimizer = optim.Adam(model.parameters(), lr=params['lr'])
    criterion = nn.MSELoss()
    memory = deque(maxlen=5000)

    gamma = params['gamma']
    epsilon = params['epsilon']
    episodes = 3
    batch_size = 64

    for ep in range(episodes):
        for task in tasks:
            state = get_state(task)
            if random.random() < epsilon:
                action = random.randint(0, 9)
            else:
                with torch.no_grad():
                    q_vals = model(state)
                    action = torch.argmax(q_vals).item()

            reward = get_reward(vm_list[action], task)
            memory.append((state, action, reward))
            if reward > 0:
                update_vm_resources(vm_list[action], task)

            if len(memory) >= batch_size:
                batch = random.sample(memory, batch_size)
                for s, a, r in batch:
                    target = r
                    output = model(s)[a]
                    loss = criterion(output, torch.tensor(target))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

    model_name = f"dqn_model_lr{params['lr']}_g{params['gamma']}_eps{params['epsilon']}.pth"
    torch.save(model.state_dict(), model_name)
    print(f"✅ Saved: {model_name}")
    results.append((params, model_name))

# Save best model for simulation (using final training config)
best_model_name = results[0][1]  # you can sort later based on success rate
with open("best_dqn_model.txt", "w") as f:
    f.write(best_model_name)
print("✅ Best model saved path written to 'best_dqn_model.txt'")
