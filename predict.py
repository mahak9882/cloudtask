# batch_predict.py

import pandas as pd
from odel import predict_vm
import json

CSV_PATH = "dataset/merged_openb_all_with_source.csv"

# --- GPU spec list ---
def get_gpu_spec_list(csv_path):
    df = pd.read_csv(csv_path, dtype=str)
    specs = set()
    for s in df["gpu_spec"].dropna():
        for g in str(s).split("|"):
            specs.add(g.strip())
    return sorted(list(specs))

# --- Convert single row to model-ready features ---
def preprocess_task(row, gpu_spec_list):
    qos_mapping = {'BestEffort': 0, 'Burstable': 1, 'Guaranteed': 2}
    phase_mapping = {'Running': 0, 'Pending': 1, 'Succeeded': 2, 'Failed': 3}

    def safe_float(val, default=0.0):
        try:
            return float(val) if pd.notna(val) and str(val).strip() != '' else default
        except:
            return default

    gpu_onehot = [0] * len(gpu_spec_list)
    raw_spec = str(row.get('gpu_spec', '')).strip()
    if raw_spec:
        for g in raw_spec.split("|"):
            g = g.strip()
            if g in gpu_spec_list:
                gpu_onehot[gpu_spec_list.index(g)] = 1

    features = [
        safe_float(row.get('cpu_milli'), 1000.0) / 1000.0,
        safe_float(row.get('memory_mib'), 1024.0) / 1024.0,
        safe_float(row.get('num_gpu'), 0),
        safe_float(row.get('gpu_milli'), 0) / 1000.0,
        float(qos_mapping.get(row.get('qos', ''), 0)),
        float(phase_mapping.get(row.get('pod_phase', ''), 0)),
        safe_float(row.get('creation_time'), 0) / 1e6,
        safe_float(row.get('deletion_time'), 0) / 1e6,
        safe_float(row.get('scheduled_time'), 0) / 1e6
    ] + gpu_onehot

    return features

# --- Batch predictor ---
def batch_predict(csv_path, output_file=None):
    df = pd.read_csv(csv_path, dtype=str)
    gpu_spec_list = get_gpu_spec_list(csv_path)

    results = []

    for i, row in df.iterrows():
        pod_name = row.get("name", f"task_{i}")

        try:
            features = preprocess_task(row, gpu_spec_list)
            result = predict_vm(features, return_extended=True)
            print(f"Task {pod_name} → {json.dumps(result)}")
            results.append({pod_name: result})
        except Exception as e:
            print(f"⚠️ Skipping {pod_name}: {str(e)}")

    if output_file:
        with open(output_file, "w") as f:
            for entry in results:
                json.dump(entry, f)
                f.write("\n")

    return results

# --- Entry ---
if __name__ == "__main__":
    batch_predict(CSV_PATH, output_file="predicted_tasks.jsonl")
