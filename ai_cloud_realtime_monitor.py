
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import time

# Simulate streaming cloud activity data
def simulate_data_stream(batch_size=100):
    normal = pd.DataFrame({
        'login_attempts': np.random.poisson(lam=2, size=batch_size),
        'file_access_frequency': np.random.normal(loc=100, scale=20, size=batch_size),
        'data_transfer_MB': np.random.normal(loc=500, scale=100, size=batch_size),
        'label': 1
    })
    anomalies = pd.DataFrame({
        'login_attempts': np.random.poisson(lam=10, size=5),
        'file_access_frequency': np.random.normal(loc=300, scale=50, size=5),
        'data_transfer_MB': np.random.normal(loc=1500, scale=300, size=5),
        'label': -1
    })
    return pd.concat([normal, anomalies], ignore_index=True).sample(frac=1).reset_index(drop=True)

# Initialize model
scaler = StandardScaler()
model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)

# Simulate initial training
print("Training model on initial dataset...")
initial_data = simulate_data_stream(500)
X_train = scaler.fit_transform(initial_data[['login_attempts', 'file_access_frequency', 'data_transfer_MB']])
model.fit(X_train)

# Simulate real-time monitoring
for i in range(3):  # simulate 3 data batches
    print(f"\n[Batch {i+1}] Monitoring cloud activity...")
    batch = simulate_data_stream()
    X_batch = scaler.transform(batch[['login_attempts', 'file_access_frequency', 'data_transfer_MB']])
    y_pred = model.predict(X_batch)

    batch['prediction'] = y_pred
    print(classification_report(batch['label'], y_pred, target_names=['Anomaly', 'Normal']))

    # Visualization
    plt.figure()
    plt.scatter(X_batch[:, 1], X_batch[:, 2], c=y_pred, cmap='coolwarm', s=10)
    plt.title(f"Batch {i+1}: Anomaly Detection")
    plt.xlabel("File Access Frequency")
    plt.ylabel("Data Transfer (MB)")
    plt.show()

    time.sleep(1)
