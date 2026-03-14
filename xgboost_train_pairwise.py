import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

df = pd.read_csv("task_vm_training_data.csv")

X = df[['cpu_milli', 'num_gpu', 'gpu_milli', 'gpu_spec', 'vm_cpu', 'vm_gpu']]
y = df['success']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = xgb.XGBClassifier(eval_metric='logloss')
model.fit(X_train, y_train)

joblib.dump(model, "xgboost_vm_selector.pkl")
print("✅ Model trained and saved")

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
