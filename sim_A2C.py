 # ---------------------------- simulation_a2c.py ----------------------------
import simpy
import csv
from MODEL_A2C import predict_vm , global_vm_list # separate module with predict_vm logic

SUCCESS_RATE_THRESHOLD = 15.0
CSV_FILE = "merged_without_nan_scheduled_time.csv"
NUM_FALLBACK_VMS = 5

class VM:
    def __init__(self, env, id, gpu_spec=None):
        self.id = id
        self.gpu_spec = gpu_spec
        self.resource = simpy.Resource(env, capacity=2)

class CloudEnvironment:
    def __init__(self, env, vms):
        self.env = env
        self.vms = vms
        self.successful_tasks = 0
        self.total_tasks = 0

    def run_task(self, task_id, features):
        required_gpu_list = features[4]
        task_name = features[10]

        result = predict_vm(features, return_extended=True)
        predicted_vm_id = result["selected_vm_id"]
        q_values = result["q_values"]
        confidence = result["confidence"]

        if not (0 <= predicted_vm_id < len(self.vms)):
            print(f"[{self.env.now:.2f}] ❌ Task {task_name} REJECTED: Invalid VM predicted")
            return

        vm = self.vms[predicted_vm_id]
        exec_time = features[9] - features[7]
        exec_time = max(exec_time, 0.01)

        with vm.resource.request() as req:
            yield req
            yield self.env.timeout(exec_time * 0.9)

        self.total_tasks += 1
        if exec_time <= SUCCESS_RATE_THRESHOLD:
            self.successful_tasks += 1
            result_str = "✅ Success"
        else:
            result_str = "⚠️ Failed"

        print(json_format(task_name, result))
        print(f"[{self.env.now:.2f}] Task {task_name} completed on VM {vm.id} --> {result_str}")

def json_format(task_id, result_dict):
    return f'Task {task_id} → {result_dict}'

def safe_float(value, default=0.0):
    try:
        return float(value) if str(value).strip() != '' else default
    except:
        return default

def load_tasks_and_gpu_types(filename):
    tasks = []
    gpu_spec_set = set()
    qos_mapping = {'BestEffort': 0, 'Burstable': 1, 'Guaranteed': 2}
    phase_mapping = {'Running': 0, 'Pending': 1, 'Succeeded': 2, 'Failed': 3}

    with open(filename, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_gpu_spec = row.get('gpu_spec', '').strip()
            parsed_gpu = [s.strip() for s in raw_gpu_spec.split('|') if s.strip()] if raw_gpu_spec else []
            for spec in parsed_gpu:
                gpu_spec_set.add(spec)

            features = [
                safe_float(row.get('cpu_milli')),
                safe_float(row.get('memory_mib')),
                safe_float(row.get('num_gpu')),
                safe_float(row.get('gpu_milli')),
                parsed_gpu,
                float(qos_mapping.get(row.get('qos', ''), -1)),
                float(phase_mapping.get(row.get('pod_phase', ''), -1)),
                safe_float(row.get('creation_time')),
                safe_float(row.get('deletion_time')),
                safe_float(row.get('scheduled_time')),
                row.get("name", f"task_{len(tasks)}")
            ]
            tasks.append(features)

    return tasks, sorted(gpu_spec_set)

def task_generator(env, cloud, task_list):
    for i, features in enumerate(task_list):
        yield env.process(cloud.run_task(i, features))
        yield env.timeout(0.5)

def run_simulation():
    env = simpy.Environment()
    task_list, gpu_specs = load_tasks_and_gpu_types(CSV_FILE)

    vms = [VM(env, i, spec) for i, spec in enumerate(gpu_specs)]
    for i in range(NUM_FALLBACK_VMS):
        vms.append(VM(env, len(vms), gpu_spec=None))

    global_vm_list.clear()
    global_vm_list.extend(vms)

    cloud = CloudEnvironment(env, vms)

    print(f"\n GPU Types Found: {gpu_specs}")
    print(f" Created {len(vms)} VMs ({len(gpu_specs)} GPU-specific + {NUM_FALLBACK_VMS} fallback)")
    print(f" Loaded {len(task_list)} tasks from '{CSV_FILE}'.")

    env.process(task_generator(env, cloud, task_list))
    env.run()

    print("\n Simulation Results")
    print(f"Total Tasks: {cloud.total_tasks}")
    print(f"Successful Tasks: {cloud.successful_tasks}")
    if cloud.total_tasks > 0:
        print(f"✅ Success Rate: {(cloud.successful_tasks / cloud.total_tasks) * 100:.2f}%")
    else:
        print(" No tasks were executed.")

if __name__ == "__main__":
    run_simulation()
