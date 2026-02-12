import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib

class DataLoader:
    def __init__(self, data_path):
        self.data_path = data_path
        self.pipeline = None
        self.numerical_cols = []
        self.categorical_cols = ['proto', 'service', 'state']
        
    def load_and_preprocess(self, train_file, test_file=None):
        print("Loading data...")
        train_df = pd.read_parquet(train_file)
        
        # Identify numerical columns (exclude targets and categoricals)
        exclude_cols = ['id', 'label', 'attack_cat'] + self.categorical_cols
        self.numerical_cols = [c for c in train_df.columns if c not in exclude_cols]
        
        # Preprocessing Pipeline
        # We need to handle unknown categories in test data, so handle_unknown='ignore'
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', MinMaxScaler(), self.numerical_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore'), self.categorical_cols)
            ]
        )
        
        self.pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
        
        # Filter Normal traffic for training
        # Assuming label 0 is normal. 'attack_cat' might be 'Normal'
        print("Filtering normal traffic for training...")
        normal_train_df = train_df[train_df['label'] == 0]
        
        # Fit pipeline on normal training data
        print("Fitting preprocessing pipeline...")
        X_train = self.pipeline.fit_transform(normal_train_df)
        
        X_test = None
        y_test = None
        
        if test_file:
             test_df = pd.read_parquet(test_file)
             X_test = self.pipeline.transform(test_df)
             y_test = test_df['label'].values
        else:
             # If no separate test file, split the training data? 
             # Usually we want to test on anomalies too. 
             # For now, let's assume we might use the remaining train data (anomalies) for testing if needed
             pass

        print(f"Training data shape: {X_train.shape}")
        return X_train, X_test, y_test

    def save_pipeline(self, filepath='preprocessor.joblib'):
        joblib.dump(self.pipeline, filepath)
        print(f"Pipeline saved to {filepath}")

    def load_pipeline(self, filepath='preprocessor.joblib'):
        self.pipeline = joblib.load(filepath)
        print(f"Pipeline loaded from {filepath}")

def sliding_window_stream(data, window_size=1, batch_size=32):
    """
    Simulates a stream of data.
    Yields batches of data.
    """
    num_samples = data.shape[0]
    for i in range(0, num_samples, batch_size):
        yield data[i:i + batch_size]
