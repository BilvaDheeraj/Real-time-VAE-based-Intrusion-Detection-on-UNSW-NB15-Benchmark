import torch
import numpy as np
import time

class InferenceEngine:
    def __init__(self, model, threshold=None):
        self.model = model
        self.threshold = threshold
        self.history = []

    def set_threshold(self, threshold):
        self.threshold = threshold

    def predict(self, batch_data):
        """
        Predicts anomaly scores for a batch of data.
        """
        self.model.eval()
        with torch.no_grad():
            tensor_data = torch.FloatTensor(batch_data)
            recon_batch, _, _ = self.model(tensor_data)
            
            # Calculate Reconstruction Error (MSE per sample)
            error = torch.mean((tensor_data - recon_batch) ** 2, dim=1).numpy()
            
        return error

    def process_stream(self, data_stream, percentiles=95):
        """
        Simulates processing a stream of data.
        """
        for batch in data_stream:
            start_time = time.time()
            error = self.predict(batch)
            inference_time = (time.time() - start_time) * 1000 # ms
            
            # Determine anomalies
            if self.threshold is None:
                # Dynamic thresholding based on initial batch? 
                # Or user should set it. For now default to mean + 2std if not set
                self.threshold = np.mean(error) + 2 * np.std(error)
            
            anomalies = error > self.threshold
            
            yield {
                'error': error,
                'anomalies': anomalies,
                'inference_time': inference_time,
                'timestamp': time.time()
            }
