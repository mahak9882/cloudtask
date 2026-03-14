#simulation
import simpy
import csv
import random
from dqn_MODEL_grid import predict_vm, global_vm_list

SUCCESS_RATE_THRESHOLD = 12
CSV_FILE = "Dataset_1.csv"
NUM_FALLBACK_VMS = 3

class VM:
    def __init__(self, env, id, gpu_spec=None):
        self.id = id
        self.gpu_spec = gpu_spec
        self.resource = simpy.Resource(env, capacity=1)

class CloudEnvironment:
    def __init__(self, env, vms):
        self.env = env
        self.vms = vms
        self.successful_tasks = 0
        self.total_tasks = 0

    def run_task(self, task_id, features):
        required_gpu_list = features[4]
        prediction_result = predict_vm(features, return_extended=True)
        predicted_vm_id = prediction_result["selected_vm_id"]
        confidence = prediction_result["confidence"]
        q_values = prediction_result["q_values"]

        if not (0 <= predicted_vm_id < len(self.vms)):
            print(f"[{self.env.now:.2f}] Task {task_id} REJECTED: Invalid VM predicted")
            return

        predicted_vm = self.vms[predicted_vm_id]

        # Choose fallback VM if GPU spec mismatches
        if not required_gpu_list:
            fallback_vms = [v for v in self.vms if v.gpu_spec is None]
            vm = random.choice(fallback_vms) if fallback_vms else predicted_vm
        elif predicted_vm.gpu_spec in required_gpu_list:
            vm = predicted_vm
        else:
            matching_vms = [v for v in self.vms if v.gpu_spec in required_gpu_list]
            vm = random.choice(matching_vms) if matching_vms else None
            if vm is None:
                print(f"[{self.env.now:.2f}] Task {task_id} REJECTED: No matching VM for GPU spec {required_gpu_list}")
                return

        exec_time = features[0] / 1000  # Convert from milli to seconds

        print(f"\nTask {task_id} → DQN Prediction")
        print(f"   ├─ Predicted VM ID : {predicted_vm_id}")
        print(f"   ├─ Assigned to     : VM {vm.id} (GPU: {vm.gpu_spec})")
        print(f"   ├─ Confidence      : {confidence}")
        print(f"   ├─ Q-values        : {q_values}")
        print(f"   └─ Execution Time  : {exec_time:.2f} seconds")

        with vm.resource.request() as req:
            yield req
            yield self.env.timeout(exec_time)

        self.total_tasks += 1
        if exec_time <= SUCCESS_RATE_THRESHOLD:
            self.successful_tasks += 1
            result = "Success"
        else:
            result = "Failed"

        print(f"Task {task_id} completed on VM {vm.id} --> {result}")

def load_tasks_and_gpu_types(filename):
    tasks = []
    gpu_spec_set = set()
    qos_mapping = {'BestEffort': 0, 'Burstable': 1, 'Guaranteed': 2}
    phase_mapping = {'Running': 0, 'Pending': 1, 'Succeeded': 2, 'Failed': 3}

    def safe_float(value, default=0.0):
        try:
            return float(value) if str(value).strip() != '' else default
        except:
            return default

    with open(filename, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_gpu_spec = row.get('gpu_spec', '').strip()
            parsed_gpu = []
            if raw_gpu_spec:
                specs = [s.strip() for s in raw_gpu_spec.split('|') if s.strip()]
                parsed_gpu = specs
                for spec in specs:
                    gpu_spec_set.add(spec)

            features = [
                safe_float(row['cpu_milli']),
                safe_float(row['memory_mib']),
                safe_float(row['num_gpu']),
                safe_float(row['gpu_milli']),
                parsed_gpu,
                float(qos_mapping.get(row['qos'], -1)),
                float(phase_mapping.get(row['pod_phase'], -1)),
                safe_float(row['creation_time']),
                safe_float(row['deletion_time']),
                safe_float(row['scheduled_time'])
            ]
            tasks.append(features)

    return tasks, sorted(gpu_spec_set)

def task_generator(env, cloud, task_list):
    for i, features in enumerate(task_list):
        yield env.process(cloud.run_task(i, features))
        yield env.timeout(1)  # Simulate time between arrivals

def run_simulation():
    env = simpy.Environment()
    task_list, gpu_specs = load_tasks_and_gpu_types(CSV_FILE)
    vms = [VM(env, i, spec) for i, spec in enumerate(gpu_specs)]
    for i in range(NUM_FALLBACK_VMS):
        vms.append(VM(env, len(vms), gpu_spec=None))

    global_vm_list.clear()
    global_vm_list.extend(vms)

    cloud = CloudEnvironment(env, vms)

    print(f"\nGPU Types Found: {gpu_specs}")
    print(f"Created {len(vms)} VMs ({len(gpu_specs)} GPU-specific + {NUM_FALLBACK_VMS} fallback)")
    print(f"Loaded {len(task_list)} tasks from '{CSV_FILE}'.")

    env.process(task_generator(env, cloud, task_list))
    env.run()

    # Final Summary
    print("\nFinal Simulation Summary")
    print(f"-----------------------------")
    print(f"Total Tasks      : {cloud.total_tasks}")
    print(f"Successful Tasks : {cloud.successful_tasks}")
    if cloud.total_tasks > 0:
        success_rate = (cloud.successful_tasks / cloud.total_tasks) * 100
        print(f"Success Rate     : {success_rate:.2f}%")
    else:
        print("No tasks were executed.")

if __name__ == "__main__":
    run_simulation()