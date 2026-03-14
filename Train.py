# train_dqn.py
import pandas as pd
import simpy
import numpy as np
from New_DQN import Task, VM, SimPySchedulerEnv
from DQN_Agent import DQNAgent

# Load the dataset
try:
    df = pd.read_csv("merged_without_nan_scheduled_time.csv", low_memory=False)
except Exception as e:
    print(f"Failed to load dataset: {e}")
    exit(1)

# Use relevant features only
if not {'cpu_milli', 'gpu_milli'}.issubset(df.columns):
    print("Missing required columns in dataset.")
    exit(1)

# Replace missing values and filter invalid entries
df = df.fillna(0)
df = df[(df['cpu_milli'] > 0) & (df['gpu_milli'] >= 0)]
task_data = df[['cpu_milli', 'gpu_milli']].values

def extract_state(task_features, vms):
    state = list(task_features)
    for vm in vms:
        state.append(vm.available_cpu)
        state.append(vm.available_gpu)
    return np.array(state, dtype=np.float32)

num_vms = 10
total_cpu = 8000
total_gpu = 10000
input_size = 2 + 2 * num_vms  # cpu + gpu of task + VM states
action_size = num_vms

episodes = 10
batch_size = 32

agent = DQNAgent(input_size, action_size)

for ep in range(episodes):
    env = simpy.Environment()
    vms = [VM(env, i, total_cpu, total_gpu) for i in range(num_vms)]
    scheduler_env = SimPySchedulerEnv(env, vms)

    for task_id, row in enumerate(task_data):
        task = Task(task_id=task_id, cpu_req=row[0], gpu_req=row[1], duration=1)
        state = extract_state(row, vms)
        action = agent.act(state)
        result = scheduler_env.step(task, action)

        if isinstance(result, tuple):
            proc, reward = result
        else:
            proc, reward = None, -1

        if proc is not None and hasattr(proc, '__iter__'):
            env.process(proc)
        env.run(until=env.now + 0.1)

        next_state = extract_state(row, vms)
        agent.remember(state, action, reward, next_state)
        agent.replay(batch_size)

    success_rate = 100 * scheduler_env.successful_tasks / scheduler_env.total_tasks if scheduler_env.total_tasks > 0 else 0
    print(f"Episode {ep+1}: Success Rate = {scheduler_env.successful_tasks}/{scheduler_env.total_tasks} ({success_rate:.2f}%)")

