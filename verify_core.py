import os
import torch
import numpy as np
from data_loader import DataLoader, sliding_window_stream
from vae_model import VAE, train_model
from inference import InferenceEngine
from metrics import calculate_metrics

def run_verification():
    print("Starting verification...")
    # Paths
    data_dir = r"c:/Users/Bilva Dheeraj/Documents/WOXSEN/3rd year/Sem-6/Deep learning/PBL/UNSW-NB15"
    train_file = os.path.join(data_dir, "UNSW_NB15_training-set.parquet")
    test_file = os.path.join(data_dir, "UNSW_NB15_testing-set.parquet")
    
    # 1. Data Loading
    print("\n[1/4] Testing Data Loading...")
    loader = DataLoader(data_dir)
    try:
        X_train, X_test, y_test = loader.load_and_preprocess(train_file, test_file)
        print(f"Data loaded successfully. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    except Exception as e:
        print(f"Data loading failed: {e}")
        return

    # 2. Model Training
    print("\n[2/4] Testing Model Training...")
    input_dim = X_train.shape[1]
    model = VAE(input_dim, latent_dim=10)
    try:
        # Train for just 1 epoch to verify it runs
        model = train_model(model, X_train, epochs=1, batch_size=256) 
        print("Model training successful.")
    except Exception as e:
        print(f"Model training failed: {e}")
        return

    # 3. Inference
    print("\n[3/4] Testing Inference...")
    engine = InferenceEngine(model)
    try:
        # Use a small subset of test data
        test_batch = X_test[:100]
        predictions = engine.predict(test_batch)
        print(f"Inference successful. Predictions mean: {np.mean(predictions):.4f}")
        
        # Set threshold
        engine.set_threshold(np.mean(predictions) + 2 * np.std(predictions))
    except Exception as e:
        print(f"Inference failed: {e}")
        return

    # 4. Metrics & Streaming
    print("\n[4/4] Testing Streaming and Metrics...")
    try:
        stream = sliding_window_stream(X_test[:200], window_size=50, batch_size=10)
        results = []
        for batch in engine.process_stream(stream):
            results.append(batch)
        
        print(f"Processed {len(results)} batches.")
        if len(results) > 0:
            print("Streaming verification passed.")
        else:
            print("Streaming yielded no results.")
            
    except Exception as e:
        print(f"Streaming failed: {e}")
        return

    print("\n✅ Verification Completed Successfully!")

if __name__ == "__main__":
    run_verification()
