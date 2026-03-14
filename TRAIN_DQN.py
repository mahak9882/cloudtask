 # train_dqn.py
import pandas as pd
import simpy
import numpy as np
from New_DQN import Task, VM, SimPySchedulerEnv
from DQN_Agent import DQNAgent

# Load the dataset
df = pd.read_csv("merged_without_nan_scheduled_time.csv", low_memory=False)

# Use relevant features only
task_data = df[['cpu_milli', 'gpu_milli', 'scheduled_time']].values

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

    for row in task_data:
        task = Task(task_id=0, cpu_req=row[0], gpu_req=row[1], duration=1)
        state = extract_state(row[:2], vms)
        action = agent.act(state)
        proc, reward = scheduler_env.step(task, action)
        env.run(until=env.now + 0.1)

        next_state = extract_state(row[:2], vms)
        agent.remember(state, action, reward, next_state)
        agent.replay(batch_size)

    print(f"Episode {ep+1}: Success Rate = {scheduler_env.successful_tasks}/{scheduler_env.total_tasks} ({100 * scheduler_env.successful_tasks / scheduler_env.total_tasks:.2f}%)")
