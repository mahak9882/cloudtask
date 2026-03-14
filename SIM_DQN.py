import simpy
import csv
import json
from TRAIN_DQN_1 import predict_vm  # Must be defined in TRAIN_DQN_1.py

SUCCESS_RATE_THRESHOLD = 15.0
CSV_FILE = "merged_without_nan_scheduled_time.csv"
NUM_VMS = 10

# --- Load dataset and extract features ---
def load_tasks(filename):
    tasks = []

    def safe_float(value, default=0.0):
        try:
            return float(value) if str(value).strip() != '' else default
        except:
            return default

    with open(filename, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            gpu_spec_raw = str(row.get("gpu_spec", '')).strip()
            parsed_gpu = [g.strip() for g in gpu_spec_raw.split('|') if g.strip()] if gpu_spec_raw else []

            features = [
                safe_float(row.get('cpu_milli')),
                safe_float(row.get('memory_mib')),
                safe_float(row.get('num_gpu')),
                safe_float(row.get('gpu_milli')),
                parsed_gpu,
                row.get('qos', ''),
                row.get('pod_phase', ''),
                safe_float(row.get('creation_time')),
                safe_float(row.get('deletion_time')),
                safe_float(row.get('scheduled_time')),
                row.get("name", f"task_{len(tasks)}")  # task ID
            ]
            tasks.append(features)

    return tasks

# --- VM class ---
class VM:
    def __init__(self, env, vm_id):
        self.id = vm_id
        self.resource = simpy.Resource(env, capacity=1)

# --- Cloud Environment ---
class CloudEnvironment:
    def __init__(self, env, vms):
        self.env = env
        self.vms = vms
        self.successful_tasks = 0
        self.total_tasks = 0
        self.vm_loads = [0.1] * len(vms)

    def run_task(self, task_id, features):
        task_name = features[10]
        result = predict_vm(features, vm_loads=self.vm_loads, return_extended=True)

        predicted_vm_id = result["selected_vm_id"]
        confidence = result["confidence"]
        q_values = result["q_values"]

        if not (0 <= predicted_vm_id < len(self.vms)):
            print(f"[{self.env.now:.2f}] ❌ Task {task_name} REJECTED: Invalid VM predicted")
            return

        vm = self.vms[predicted_vm_id]
        exec_time = max(features[9] - features[7], 0.01)

        self.vm_loads[predicted_vm_id] += exec_time / 100.0  # simulate load

        with vm.resource.request() as req:
            yield req
            yield self.env.timeout(exec_time)

        self.vm_loads[predicted_vm_id] -= exec_time / 100.0  # release load

        self.total_tasks += 1
        if exec_time <= SUCCESS_RATE_THRESHOLD:
            self.successful_tasks += 1
            status = "✅ Success"
        else:
            status = "⚠️ Failed"

        print(json.dumps({
            "task": task_name,
            "vm": predicted_vm_id,
            "confidence": confidence,
            "status": status,
            "q_values": q_values
        }))

# --- Task Generator ---
def task_generator(env, cloud, task_list):
    for i, features in enumerate(task_list):
        yield env.process(cloud.run_task(i, features))
        yield env.timeout(1)  # small gap between task submissions

# --- Main simulation runner ---
def run_simulation():
    env = simpy.Environment()
    tasks = load_tasks(CSV_FILE)
    vms = [VM(env, i) for i in range(NUM_VMS)]

    cloud = CloudEnvironment(env, vms)

    print(f"\n📦 Loaded {len(tasks)} tasks from '{CSV_FILE}'")
    print(f"🖥️  Created {len(vms)} VMs\n")

    env.process(task_generator(env, cloud, tasks))
    env.run()

    print("\n📊 Simulation Results:")
    print(f"Total Tasks: {cloud.total_tasks}")
    print(f"Successful Tasks: {cloud.successful_tasks}")
    if cloud.total_tasks > 0:
        print(f"✅ Success Rate: {(cloud.successful_tasks / cloud.total_tasks) * 100:.2f}%")
    else:
        print("⚠️ No tasks were executed.")

# --- Entry point ---
if __name__ == "__main__":
    run_simulation()
