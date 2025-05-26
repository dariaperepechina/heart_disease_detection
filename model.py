import pandas as pd
import numpy as np
import joblib
import os
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

class ModelTrainer:
    def __init__(self, model_dir='models'):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
    def train(self, train_path, val_path=None, model_type='logistic'):
        print(f"Training {model_type} model...")
        
        train_df = pd.read_csv(train_path)
        
        target_col = train_df.columns[-1]
        
        X_train = train_df.drop(columns=[target_col])
        y_train = train_df[target_col]
        
        if val_path:
            val_df = pd.read_csv(val_path)
            X_val = val_df.drop(columns=[target_col])
            y_val = val_df[target_col]
        
        unique_values = np.unique(y_train)
        print(f"Target unique values: {unique_values}")
        print(f"Number of unique values: {len(unique_values)}")
        
        is_classification = len(unique_values) <= 10
        task_type = "classification" if is_classification else "regression"
        print(f"Detected task type: {task_type}")
        
        if is_classification:
            encoder = LabelEncoder()
            y_train = encoder.fit_transform(y_train)
            if val_path:
                y_val = encoder.transform(y_val)
                
            encoder_path = os.path.join(self.model_dir, "label_encoder.pkl")
            joblib.dump(encoder, encoder_path)
            print(f"Label encoder saved to {encoder_path}")
            print(f"Encoded target unique values: {np.unique(y_train)}")
        
        if model_type == 'logistic':
            if is_classification:
                model = LogisticRegression(max_iter=1000, random_state=42)
            else:
                model = LinearRegression()
        elif model_type == 'random_forest':
            if is_classification:
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        model.fit(X_train, y_train)
        
        model_path = os.path.join(self.model_dir, f"{model_type}_{task_type}_model.pkl")
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")
        
        y_train_pred = model.predict(X_train)
        
        if is_classification:
            self._evaluate_classification(y_train, y_train_pred, "Training")
            
            if val_path:
                y_val_pred = model.predict(X_val)
                self._evaluate_classification(y_val, y_val_pred, "Validation")
                
                with open(os.path.join(self.model_dir, f"{model_type}_evaluation.txt"), 'w') as f:
                    f.write(f"Model: {model_type} (Classification)\n")
                    f.write("Classification Report:\n")
                    f.write(classification_report(y_val, y_val_pred))
                    
                    f.write("\nClass mapping:\n")
                    for i, label in enumerate(encoder.classes_):
                        f.write(f"Class {i}: {label}\n")
        else:
            self._evaluate_regression(y_train, y_train_pred, "Training")
            
            if val_path:
                y_val_pred = model.predict(X_val)
                self._evaluate_regression(y_val, y_val_pred, "Validation")
                
                with open(os.path.join(self.model_dir, f"{model_type}_evaluation.txt"), 'w') as f:
                    f.write(f"Model: {model_type} (Regression)\n")
                    f.write(f"Training RMSE: {np.sqrt(mean_squared_error(y_train, y_train_pred)):.4f}\n")
                    f.write(f"Training MAE: {mean_absolute_error(y_train, y_train_pred):.4f}\n")
                    f.write(f"Training R²: {r2_score(y_train, y_train_pred):.4f}\n\n")
                    f.write(f"Validation RMSE: {np.sqrt(mean_squared_error(y_val, y_val_pred)):.4f}\n")
                    f.write(f"Validation MAE: {mean_absolute_error(y_val, y_val_pred):.4f}\n")
                    f.write(f"Validation R²: {r2_score(y_val, y_val_pred):.4f}\n")
        
        return model
    
    def _evaluate_classification(self, y_true, y_pred, dataset_name):
        """
        Evaluate classification model.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            dataset_name: Name of the dataset (e.g., "Training", "Validation")
        """
        accuracy = accuracy_score(y_true, y_pred)
        print(f"{dataset_name} accuracy: {accuracy:.4f}")
        
        if len(np.unique(y_true)) == 2:  
            try:
                precision = precision_score(y_true, y_pred)
                recall = recall_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred)
                
                print(f"{dataset_name} precision: {precision:.4f}")
                print(f"{dataset_name} recall: {recall:.4f}")
                print(f"{dataset_name} F1 score: {f1:.4f}")
            except Exception as e:
                print(f"Error calculating binary metrics: {str(e)}")
    
    def _evaluate_regression(self, y_true, y_pred, dataset_name):
        """
        Evaluate regression model.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            dataset_name: Name of the dataset (e.g., "Training", "Validation")
        """
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        print(f"{dataset_name} RMSE: {rmse:.4f}")
        print(f"{dataset_name} MAE: {mae:.4f}")
        print(f"{dataset_name} R²: {r2:.4f}")
