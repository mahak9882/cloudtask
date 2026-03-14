# ---------------------------- supervised_model.py ----------------------------
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# --- Constants ---
NUM_VMS = 10
GPU_TYPES = ['CPU-only', 'T4', 'V100', 'A100', 'V100M16', 'V100M32', 'P100', 'G2', 'G3']
gpu_index = {gpu: i for i, gpu in enumerate(GPU_TYPES)}
STATE_DIM = 6 + len(GPU_TYPES)

# --- GPU One-Hot Encoding ---
def gpu_spec_to_onehot(gpu_spec):
    vec = [0] * len(GPU_TYPES)
    key = gpu_spec if gpu_spec in gpu_index else 'CPU-only'
    vec[gpu_index[key]] = 1
    return vec

# --- Load and Prepare Data ---
def load_supervised_data(csv_path):
    df = pd.read_csv(csv_path, low_memory=False)
    df.fillna(0, inplace=True)

    qos_map = {'BestEffort': 0, 'Burstable': 1, 'Guaranteed': 2}
    phase_map = {'Running': 0, 'Pending': 1, 'Succeeded': 2, 'Failed': 3}

    features = []
    labels = []

    for _, row in df.iterrows():
        try:
            exec_time = float(row['deletion_time']) - float(row['creation_time'])
            exec_time = max(exec_time, 0.01)
        except:
            exec_time = 1.0

        gpu_spec = str(row.get('gpu_spec', '')).strip()
        gpu = gpu_spec if gpu_spec in gpu_index else 'CPU-only'

        state = [
            exec_time,
            15.0,
            0.1, 0.1, 0.1,  # Dummy VM loads
            1.0 if gpu != 'CPU-only' else 0.0,
            *gpu_spec_to_onehot(gpu)
        ]

        label = np.random.randint(0, NUM_VMS)  # You can replace with smarter logic
        features.append(state)
        labels.append(label)

    return np.array(features), np.array(labels)

# --- Simple Feedforward Classifier ---
class SupervisedVMClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SupervisedVMClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

# --- Training Script ---
def train_supervised_model(csv_file):
    X, y = load_supervised_data(csv_file)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = SupervisedVMClassifier(X.shape[1], NUM_VMS)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(20):
        model.train()
        inputs = torch.tensor(X_train, dtype=torch.float32)
        targets = torch.tensor(y_train, dtype=torch.long)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}/20 - Loss: {loss.item():.4f}")

    # Evaluation
    model.eval()
    with torch.no_grad():
        test_inputs = torch.tensor(X_test, dtype=torch.float32)
        test_outputs = model(test_inputs)
        predictions = torch.argmax(test_outputs, dim=1).numpy()

        print("\nClassification Report:")
        print(classification_report(y_test, predictions))
        print(f"Accuracy: {accuracy_score(y_test, predictions)*100:.2f}%")

    # Save the model
    torch.save(model.state_dict(), "supervised_vm_model.pth")
    print("\n✅ Supervised model trained and saved as 'supervised_vm_model.pth'")

if __name__ == "__main__":
    train_supervised_model("merged_without_nan_scheduled_time.csv")
