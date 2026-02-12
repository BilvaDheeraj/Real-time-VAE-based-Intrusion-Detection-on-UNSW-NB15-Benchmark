# Real-Time Anomaly Detection System with VAE

## Project Overview
This project implements a real-time anomaly detection system using Variational Autoencoders (VAEs) trained on the UNSW-NB15 dataset. It features a live dashboard built with Streamlit for monitoring network traffic anomalies.

## Setup Instructions

1.  **Environment**: Ensure Python 3.8+ is installed.
2.  **Dependencies**: Install required packages:
    ```bash
    pip install pandas numpy scikit-learn torch streamlit altair pyarrow
    ```

## Running the Application

1.  Navigate to the project directory:
    ```bash
    cd "c:/Users/Bilva Dheeraj/Documents/WOXSEN/3rd year/Sem-6/Deep learning/PBL"
    ```

2.  Run the Streamlit Dashboard:
    ```bash
    # Option 1 (Recommended)
    python -m streamlit run dashboard.py

    # Option 2 (If streamlit is in your PATH)
    streamlit run dashboard.py
    ```

3.  **Using the Dashboard**:
    - Select the dataset directory (default is correct).
    - Adjust simulation settings (Window Size, Stride, Speed).
    - Adjust model parameters (Latent Dimension, Epochs).
    - Click **"Train New Model"** to train the VAE on normal traffic.
    - Once trained, click **"Start Live Simulation"** to begin real-time anomaly detection.

## Project Structure
- `data_loader.py`: Handles loading and preprocessing of Parquet data.
- `vae_model.py`: PyTorch implementation of the VAE.
- `inference.py`: Real-time inference engine.
- `metrics.py`: Evaluation metrics.
- `dashboard.py`: Streamlit frontend.
- `verify_core.py`: Script to verifying backend logic (non-GUI).
