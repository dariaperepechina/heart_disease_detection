import pandas as pd
import numpy as np
import joblib
import os

class Predictor:
    def __init__(self, model_path, encoder_path=None):
        self.model = joblib.load(model_path)
        self.encoder = None
        if encoder_path and os.path.exists(encoder_path):
            self.encoder = joblib.load(encoder_path)
    
    def predict(self, data_path, output_path=None):
        if isinstance(data_path, str):
            data = pd.read_csv(data_path)
        else:
            data = data_path.copy()
        
        has_target = False
        if data.columns[-1] == 'myocardial.infarction':
            has_target = True
            target = data['myocardial.infarction']
            features = data.drop(columns=['myocardial.infarction'])
        else:
            features = data
        
        predictions = self.model.predict(features)
        
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(features)
        else:
            probabilities = None
        
        if self.encoder is not None:
            original_predictions = self.encoder.inverse_transform(predictions)
        else:
            original_predictions = predictions
        
        results = pd.DataFrame({
            'predicted_class': predictions,
            'original_label': original_predictions
        })
        
        if probabilities is not None and probabilities.shape[1] == 2:
            results['probability_class_0'] = probabilities[:, 0]
            results['probability_class_1'] = probabilities[:, 1]
        
        if has_target:
            if self.encoder is not None:
                true_encoded = self.encoder.transform(target)
                results['true_class'] = true_encoded
            else:
                results['true_class'] = target
        
        if output_path:
            results.to_csv(output_path, index=False)
            print(f"Results are in {output_path}")
        
        return results
