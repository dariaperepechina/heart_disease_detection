import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

class ModelEvaluator:
    def __init__(self, output_dir='evaluation'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def evaluate(self, model_path, test_path, encoder_path=None):
        print(f"Evaluating model from {model_path} on {test_path}")
        
        model = joblib.load(model_path)
        test_df = pd.read_csv(test_path)
        
        target_col = test_df.columns[-1]
        
        X_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col]
        
        if encoder_path and os.path.exists(encoder_path):
            encoder = joblib.load(encoder_path)
            y_test = encoder.transform(y_test)
            print(f"Transformed target using encoder from {encoder_path}")
        
        y_pred = model.predict(X_test)
        
        is_classifier = hasattr(model, 'predict_proba')
        if is_classifier:
            try:
                y_pred_proba = model.predict_proba(X_test)
                has_probas = True
            except:
                has_probas = False
        else:
            has_probas = False
        
        if is_classifier:
            self._evaluate_classifier(y_test, y_pred, y_pred_proba if has_probas else None)
        else:
            self._evaluate_regressor(y_test, y_pred)
    
    def _evaluate_classifier(self, y_true, y_pred, y_pred_proba=None):
        """Evaluate a classification model"""
        accuracy = accuracy_score(y_true, y_pred)
        print(f"Accuracy: {accuracy:.4f}")
        
        print("\nClassification Report:")
        report = classification_report(y_true, y_pred)
        print(report)
        
        with open(os.path.join(self.output_dir, 'classification_report.txt'), 'w') as f:
            f.write(f"Accuracy: {accuracy:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(report)
        
        cm = confusion_matrix(y_true, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        
        classes = np.unique(y_true)
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)
        
        thresh = cm.max() / 2
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'confusion_matrix.png'))
        
        if y_pred_proba is not None and len(np.unique(y_true)) == 2:
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic')
            plt.legend(loc="lower right")
            plt.savefig(os.path.join(self.output_dir, 'roc_curve.png'))
    
    def _evaluate_regressor(self, y_true, y_pred):
        """Evaluate a regression model"""
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"Root Mean Squared Error: {rmse:.4f}")
        print(f"Mean Absolute Error: {mae:.4f}")
        print(f"R² Score: {r2:.4f}")
        
        with open(os.path.join(self.output_dir, 'regression_metrics.txt'), 'w') as f:
            f.write(f"Mean Squared Error: {mse:.4f}\n")
            f.write(f"Root Mean Squared Error: {rmse:.4f}\n")
            f.write(f"Mean Absolute Error: {mae:.4f}\n")
            f.write(f"R² Score: {r2:.4f}\n")
        
        plt.figure(figsize=(8, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=2)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Actual vs Predicted Values')
        plt.savefig(os.path.join(self.output_dir, 'actual_vs_predicted.png'))
