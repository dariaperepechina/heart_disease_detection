import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

class DataSplitter:
    def __init__(self, random_state=42):
        """
        Initialize the data splitter.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
    
    def split(self, data_path, train_path, test_path, val_path=None, test_size=0.2, val_size=0.1):
        """
        Split the dataset into train, test, and optionally validation sets.
        
        Args:
            data_path: Path to the processed data file
            train_path: Path to save the training data
            test_path: Path to save the test data
            val_path: Path to save the validation data (optional)
            test_size: Proportion of data to use for testing
            val_size: Proportion of data to use for validation
            
        Returns:
            Tuple of DataFrames (train, test, val)
        """
        print(f"Splitting data from {data_path}")
        
        df = pd.read_csv(data_path)
        
        target_col = None
        if 'target' in df.columns:
            target_col = 'target'
        else:
            for col in df.columns:
                if df[col].nunique() == 2:
                    target_col = col
                    print(f"Using {col} as the target column")
                    break
        
        if not target_col:
            print("Warning: No target column identified. Using the last column as target.")
            target_col = df.columns[-1]
        
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        test_df = pd.concat([X_test, y_test], axis=1)
        
        test_df.to_csv(test_path, index=False)
        print(f"Test data saved to {test_path}")
        
        if val_path:
            adjusted_val_size = val_size / (1 - test_size)
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val, test_size=adjusted_val_size, 
                random_state=self.random_state, stratify=y_train_val
            )
            
            train_df = pd.concat([X_train, y_train], axis=1)
            val_df = pd.concat([X_val, y_val], axis=1)
            
            val_df.to_csv(val_path, index=False)
            print(f"Validation data saved to {val_path}")
        else:
            train_df = pd.concat([X_train_val, y_train_val], axis=1)
        
        train_df.to_csv(train_path, index=False)
        print(f"Training data saved to {train_path}")
        
        print(f"Data split complete:")
        print(f"  Total samples: {len(df)}")
        print(f"  Training samples: {len(train_df)}")
        if val_path:
            print(f"  Validation samples: {len(val_df)}")
        print(f"  Test samples: {len(test_df)}")
        
        if val_path:
            return train_df, test_df, val_df
        else:
            return train_df, test_df
