import streamlit as st
import pandas as pd
import numpy as np
import time
import altair as alt
from data_loader import DataLoader, sliding_window_stream
from vae_model import VAE, train_model, loss_function
from inference import InferenceEngine
import torch
import os

# Set page config
st.set_page_config(
    page_title="NB15 Anomaly Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium look
st.markdown("""
<style>
    .reportview-container {
        background: #0e1117;
    }
    .main {
        background: #0e1117;
    }
    h1, h2, h3 {
        color: #ffffff;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .stMetric {
        background-color: #262730;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 20px;
        border: none;
        padding: 10px 24px;
        transition: all 0.3s ease 0s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box_shadow: 0 8px 15px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Title and Description
col1, col2 = st.columns([3, 1])
with col1:
    st.title("🛡️ Network Anomaly Detection System")
    st.markdown("### Real-time VAE-based Intrusion Detection on UNSW-NB15 Benchmark")
with col2:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/1200px-Python-logo-notext.svg.png", width=100)

# Sidebar Configuration
st.sidebar.header("Configuration Panel")
data_path = st.sidebar.text_input("Dataset Directory", "UNSW-NB15")
train_file = os.path.join(data_path, "UNSW_NB15_training-set.parquet")
test_file = os.path.join(data_path, "UNSW_NB15_testing-set.parquet")

# Simulation Parameters
st.sidebar.subheader("Simulation Settings")
window_size = st.sidebar.slider("Window Size", 10, 100, 50)
stride = st.sidebar.slider("Stride", 1, 50, 10)
speed = st.sidebar.slider("Simulation Speed (ms)", 10, 1000, 100)

# Model Parameters
st.sidebar.subheader("Model Parameters")
latent_dim = st.sidebar.slider("Latent Dimension", 2, 20, 10)
epochs = st.sidebar.number_input("Training Epochs", value=5)
train_btn = st.sidebar.button("Train New Model")

# Initialize Session State
if 'model' not in st.session_state:
    st.session_state.model = None
if 'threshold' not in st.session_state:
    st.session_state.threshold = 0.05
if 'data_loader' not in st.session_state:
    st.session_state.data_loader = DataLoader(data_path)
if 'X_test' not in st.session_state:
    st.session_state.X_test = None

# Main Content Area
placeholder = st.empty()

# Training Logic
if train_btn:
    with st.spinner("Loading and Preprocessing Data..."):
        try:
            X_train, X_test, y_test = st.session_state.data_loader.load_and_preprocess(train_file, test_file)
            st.session_state.X_test = X_test # Store for streaming
            
            input_dim = X_train.shape[1]
            model = VAE(input_dim, latent_dim)
            
            st.info(f"Training VAE on {X_train.shape[0]} normal samples...")
            model = train_model(model, X_train, epochs=epochs)
            
            st.session_state.model = model
            st.session_state.model = model
            st.success(f"Model Training Completed! (Epochs: {epochs}, Latent Dim: {latent_dim})")
            
            # Determine Threshold (95th percentile on train set reconstruction)
            model.eval()
            with torch.no_grad():
                train_tensor = torch.FloatTensor(X_train)
                recon, _, _ = model(train_tensor)
                train_loss = torch.mean((train_tensor - recon) ** 2, dim=1).numpy()
                st.session_state.threshold = np.percentile(train_loss, 95)
                st.info(f"Threshold set to {st.session_state.threshold:.4f} (95th percentile)")

        except Exception as e:
            st.error(f"Error during training: {e}")

# Live Dashboard
if st.session_state.model is not None and st.session_state.X_test is not None:
    start_simulation = st.checkbox("Start Live Simulation", value=False)
    
    if start_simulation:
        inference_engine = InferenceEngine(st.session_state.model, st.session_state.threshold)
        
        # Prepare Streaming
        # Prepare Streaming
        # Use stride and window_size from sidebar
        stream = sliding_window_stream(st.session_state.X_test, window_size=window_size, stride=stride)
        
        # Containers for plots
        chart_window = st.empty()
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        log_col = st.container()
        
        loss_history = []
        anomaly_history = []
        
        for batch_idx, batch_data in enumerate(stream):
            if not start_simulation:
                break
                
            prediction = inference_engine.predict(batch_data)
            
            # Average error for the batch to simplify plotting
            avg_error = np.mean(prediction)
            is_anomaly = avg_error > st.session_state.threshold
            
            loss_history.append(avg_error)
            anomaly_history.append(1 if is_anomaly else 0)
            
            # Update Metrics
            with metrics_col1:
                st.metric("Current Reconstruction Error", f"{avg_error:.4f}", delta_color="inverse")
            with metrics_col2:
                status = "CRITICAL" if is_anomaly else "NORMAL"
                st.metric("System Status", status)
            with metrics_col3:
                st.metric("Total Anomalies Detected", sum(anomaly_history))

            # Update Chart
            chart_data = pd.DataFrame({
                'Time': range(len(loss_history)),
                'Reconstruction Error': loss_history,
                'Threshold': [st.session_state.threshold] * len(loss_history)
            })
            
            c = alt.Chart(chart_data).mark_line().encode(
                x='Time',
                y='Reconstruction Error',
                color=alt.value("#00FFAA")
            ).properties(height=300)
            
            c_threshold = alt.Chart(chart_data).mark_line(strokeDash=[5,5]).encode(
                x='Time',
                y='Threshold',
                color=alt.value("#FF4B4B")
            )
            
            chart_window.altair_chart(c + c_threshold, use_container_width=True)
            
            # Log
            if is_anomaly:
                with log_col:
                    st.error(f"Anomaly Detected at Step {len(loss_history)}! Error: {avg_error:.4f}")
            
            time.sleep(speed / 1000)
            
else:
    st.info("Please train the model to start the simulation.")
