# 🛡️ Real-Time VAE-based Intrusion Detection System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B)](https://streamlit.io/)
[![Dataset](https://img.shields.io/badge/Dataset-UNSW--NB15-green)](https://research.unsw.edu.au/projects/unsw-nb15-dataset)

## 📌 Project Overview

This project implements a **Real-Time Anomaly Detection System** specifically designed for network security. Utilizing **Variational Autoencoders (VAEs)** built with PyTorch, it learns the underlying distribution of normal network traffic based on the robust **UNSW-NB15 Benchmark Dataset**. 

The system features an interactive, real-time **Streamlit Dashboard** that allows users to monitor network traffic streams, train the model dynamically, adjust simulation configurations, and visualize anomalies (intrusions) as they occur.

## ✨ Key Features

- **Live Stream Simulation**: Test the model's performance in real-time with an adjustable sliding window data stream.
- **Deep Learning Core**: Custom PyTorch VAE implementation capable of deep feature extraction and efficient reconstruction.
- **Dynamic Thresholding**: Automatically calculates anomaly thresholds based on the 95th percentile of reconstruction errors on normal data.
- **Interactive UI**: A sleek, premium dark-themed Streamlit dashboard for model training, parameter tuning, and real-time visualization of network health.
- **Parquet Support**: Utilizes highly optimized `parquet` files for rapid data loading and processing.

## 📂 Project Structure

```text
📁 PBL
│
├── 📄 dashboard.py         # Main entry point for the Streamlit GUI/Dashboard
├── 📄 data_loader.py       # Handles Parquet data loading, preprocessing, and streaming logic
├── 📄 vae_model.py         # PyTorch definition of the autoencoder and training scripts
├── 📄 inference.py         # Real-time inference engine processing incoming streams
├── 📄 metrics.py           # Evaluation functions
├── 📄 verify_core.py       # Standalone backend logic testing script without GUI
├── 📄 inspect_parquet.py   # Utility to inspect dataset formats
└── 📁 UNSW-NB15/           # Directory intended for the UNSW-NB15 dataset (.parquet files)
```

## 🛠️ Setup & Installation

### 1. Prerequisites
Ensure you have **Python 3.8+** installed on your system.

### 2. Install Dependencies
It is highly recommended to use a virtual environment. Install all required dependencies using `pip`:

```bash
pip install pandas numpy scikit-learn torch streamlit altair pyarrow joblib
```

### 3. Dataset Configuration
Ensure the UNSW-NB15 dataset files are placed in the `UNSW-NB15` folder within the project root. The system specifically expects:
- `UNSW_NB15_training-set.parquet`
- `UNSW_NB15_testing-set.parquet`

## 🚀 Running the Application

### Accessing the Live Dashboard

1. Navigate to the project directory in your terminal:
   ```bash
   cd "path/to/PBL"
   ```

2. Run the Streamlit application:
   ```bash
   streamlit run dashboard.py
   ```

### Dashboard Usage Guide
1. **Configure Data**: Validate the `Dataset Directory` path matches where your `.parquet` copies are held. 
2. **Set Parameters**: Adjust Model settings like `Latent Dimension` and `Training Epochs` in the dashboard sidebar.
3. **Train Model**: Click **Train New Model**. This will filter normal traffic, fit preprocessing pipelines, and train the VAE.
4. **Simulation**: Adjust `Window Size`, `Stride`, and `Simulation Speed` to configure the stream.
5. **Monitor Live**: Check **Start Live Simulation** to begin feeding the testing data into the inference engine. Watch the live Altair graphs spike and system status shift to **CRITICAL** when reconstruction errors exceed thresholds.

## 🔬 Core Technologies
- **Data Engineering**: `pandas`, `scikit-learn` (Pipelines, MinMaxScaler, OneHotEncoder), `pyarrow`
- **Machine Learning**: `PyTorch` (`torch.nn`, `torch.optim`)
- **Frontend / Visualization**: `Streamlit`, `Altair`

## 📜 Dataset Reference
**UNSW-NB15 Dataset**: Created by the Cyber Range Lab of UNSW Canberra. It provides a comprehensive mix of normal activities and modern synthetic contemporary attack behaviors.
