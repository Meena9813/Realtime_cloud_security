
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
import plotly.express as px
import datetime

st.set_page_config(page_title="AI Cloud Threat Monitor", layout="wide", page_icon="🛡️")

st.markdown("""
    <style>
        .main-header {
            text-align: center;
            background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
            padding: 30px;
            border-radius: 10px;
            color: white;
        }
        .main-header h1 {
            font-size: 3em;
            margin-bottom: 0;
        }
        .main-header p {
            font-size: 1.1em;
            margin-top: 5px;
            color: #d1ecf1;
        }
        .footer {
            text-align: center;
            font-size: 0.9em;
            padding: 10px;
            margin-top: 30px;
            color: #aaa;
        }
        .metric-box {
            background-color: #f4f4f4;
            padding: 12px;
            border-radius: 12px;
            box-shadow: 0px 3px 10px rgba(0,0,0,0.1);
            text-align: center;
        }
    </style>
    <div class="main-header">
        <h1>🛡️ AI Cloud Threat Detection Dashboard</h1>
        <p>Real-Time Anomaly Detection for E-Commerce Cloud Activity</p>
    </div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
    <style>
        .sidebar-container {
            background-color: #f0f8ff;
            padding: 20px 10px;
            border-radius: 10px;
            box-shadow: inset 0 0 5px #ccc;
        }
        .sidebar-container h2 {
            font-size: 1.4em;
            color: #333;
        }
    </style>
    <div class="sidebar-container">
        <h2>🔧 Configuration Settings</h2>
        <p>Adjust your monitoring setup below:</p>
    </div>
""", unsafe_allow_html=True)

batch_size = st.sidebar.slider("Batch Size", 100, 1000, 300, step=100)
contamination = st.sidebar.slider("Anomaly Contamination (%)", 0.01, 0.20, 0.05, step=0.01)
simulate_batches = st.sidebar.slider("Number of Batches", 1, 5, 3)
start_monitoring = st.sidebar.button("🚀 Start Real-Time Detection")

# Anomaly Explanation Guide
with st.expander("ℹ️ What Counts as Unusual Activity?"):
    st.markdown("""
    This system uses **AI anomaly detection** to flag unusual behavior in cloud activity based on patterns.
    
    | Field                 | Unusual Value                        | Explanation |
    |----------------------|--------------------------------------|-------------|
    | Login Attempts       | > 7 per user                         | Could indicate brute-force |
    | Failed Logins        | > 2 in a batch                       | Abnormal authentication failures |
    | File Access Frequency| > 300 times                          | Likely script or data extraction |
    | Data Transfer (MB)   | > 1000 MB                            | Large downloads or suspicious export |
    | File Accessed        | Sudden access to backups/configs     | May be sensitive file targeting |

    These thresholds are based on **statistical deviation** from normal behavior using the Isolation Forest algorithm.
    """)


file_options = ["invoice.pdf", "user_data.csv", "product_list.json", "db_backup.sql", "config.yaml", "order_logs.txt", "session.log"]

def simulate_data(batch_size, contamination_level):
    normal = pd.DataFrame({
        'login_attempts': np.random.poisson(lam=2, size=batch_size),
        'failed_logins': np.random.binomial(n=1, p=0.1, size=batch_size),
        'file_access_frequency': np.random.normal(loc=100, scale=20, size=batch_size),
        'data_transfer_MB': np.random.normal(loc=500, scale=100, size=batch_size),
        'file_accessed': np.random.choice(file_options, size=batch_size),
        'label': 1
    })
    anomalies = pd.DataFrame({
        'login_attempts': np.random.poisson(lam=10, size=int(batch_size * contamination_level)),
        'failed_logins': np.random.binomial(n=3, p=0.5, size=int(batch_size * contamination_level)),
        'file_access_frequency': np.random.normal(loc=300, scale=50, size=int(batch_size * contamination_level)),
        'data_transfer_MB': np.random.normal(loc=1500, scale=300, size=int(batch_size * contamination_level)),
        'file_accessed': np.random.choice(file_options, size=int(batch_size * contamination_level)),
        'label': -1
    })
    return pd.concat([normal, anomalies], ignore_index=True).sample(frac=1).reset_index(drop=True)

if start_monitoring:
    tabs = st.tabs([f"📦 Batch {i+1}" for i in range(simulate_batches)])
    for batch_num in range(simulate_batches):
        with tabs[batch_num]:
            data = simulate_data(batch_size, contamination)
            features = ['login_attempts', 'failed_logins', 'file_access_frequency', 'data_transfer_MB']
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(data[features])
            model = IsolationForest(n_estimators=100, contamination=contamination, random_state=42)
            data['prediction'] = model.fit_predict(X_scaled)
            data['Status'] = data['prediction'].map({1: 'Normal ✅', -1: 'Anomaly ⚠️'})

            precision = precision_score(data["label"], data["prediction"])
            recall = recall_score(data["label"], data["prediction"])
            f1 = f1_score(data["label"], data["prediction"])

            col1, col2, col3, col4, col5 = st.columns(5)
            col1.markdown(f"<div class='metric-box'><h4>🟢 Normal</h4><p>{(data['Status'] == 'Normal ✅').sum()}</p></div>", unsafe_allow_html=True)
            col2.markdown(f"<div class='metric-box'><h4>🔴 Anomalies</h4><p>{(data['Status'] == 'Anomaly ⚠️').sum()}</p></div>", unsafe_allow_html=True)
            col3.markdown(f"<div class='metric-box'><h4>🎯 Precision</h4><p>{precision:.2f}</p></div>", unsafe_allow_html=True)
            col4.markdown(f"<div class='metric-box'><h4>📈 Recall</h4><p>{recall:.2f}</p></div>", unsafe_allow_html=True)
            col5.markdown(f"<div class='metric-box'><h4>💡 F1 Score</h4><p>{f1:.2f}</p></div>", unsafe_allow_html=True)

            fig = px.scatter(
                data, x='file_access_frequency', y='data_transfer_MB',
                color='Status', title="🔍 File Access vs. Data Transfer (MB)",
                labels={'file_access_frequency': 'Access Frequency', 'data_transfer_MB': 'Data Transfer (MB)'},
                template='plotly_dark', height=450
            )
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("📄 View Activity Log"):
                st.dataframe(data, use_container_width=True)

st.markdown(f'''
    <div class="footer">
        📅 Updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Developed for Capstone Project 2025 🎓
    </div>
''', unsafe_allow_html=True)
