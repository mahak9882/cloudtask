import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib

# Load and preprocess
df = pd.read_csv("Dataset_1.csv", low_memory=False)

# Label: 1 if scheduled_time > 0, else 0
df['success'] = df['scheduled_time'].apply(lambda x: 1 if x > 0 else 0)

# Convert gpu_spec to categorical (encoded as int)
le = LabelEncoder()
df['gpu_spec'] = le.fit_transform(df['gpu_spec'].astype(str))

# Select features
X = df[['cpu_milli', 'num_gpu', 'gpu_milli', 'gpu_spec']]
y = df['success']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Save
joblib.dump(model, "xgboost_vm_model.pkl")
print("✅ Model saved as xgboost_vm_model.pkl")

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

