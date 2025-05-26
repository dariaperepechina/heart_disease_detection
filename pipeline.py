import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from preprocessing import DataPreprocessor
from nlp_features import NLPFeatureExtractor
from model_builder import ModelBuilder
from deep_learning_model import DeepLearningModel

class Pipeline:
    def __init__(self, config=None):
        """
        Initialize the pipeline.
        
        Args:
            config: Dictionary with configuration parameters
        """
        self.config = config or {}
        
        self.numerical_columns = self.config.get('numerical_columns', [])
        self.categorical_columns = self.config.get('categorical_columns', [])
        self.text_columns = self.config.get('text_columns', [])
        self.target_column = self.config.get('target_column', None)
        
        self.output_dir = self.config.get('output_dir', 'results')
        os.makedirs(self.output_dir, exist_ok=True)
    
    def run(self, data_path):
        """
        Run the full pipeline.
        
        Args:
            data_path: Path to the data file
            
        Returns:
            Dictionary with results
        """
        print("Starting pipeline...")
        
        print("Loading data...")
        data = self._load_data(data_path)
        
        print("Preprocessing data...")
        X_train, X_val, X_test, y_train, y_val, y_test = self._preprocess_data(data)
        
        if self.text_columns:
            print("Extracting NLP features...")
            X_train, X_val, X_test = self._extract_nlp_features(X_train, X_val, X_test)
        
        print("Training traditional model...")
        traditional_model, traditional_metrics = self._train_traditional_model(X_train, X_val, y_train, y_val)
        
        dl_model = None
        dl_metrics = None
        if self.text_columns:
            print("Training deep learning model...")
            dl_model, dl_metrics = self._train_deep_learning_model(X_train, X_val, y_train, y_val)
        
        print("Evaluating models...")
        test_results = self._evaluate_models(traditional_model, dl_model, X_test, y_test)
        
        print("Saving results...")
        self._save_results(traditional_model, dl_model, test_results)
        
        print("Pipeline completed!")
        
        return {
            'traditional_model': traditional_model,
            'dl_model': dl_model,
            'test_results': test_results
        }
    
    def _load_data(self, data_path):
        """
        Load data from file.
        
        Args:
            data_path: Path to the data file
            
        Returns:
            Loaded DataFrame
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        file_ext = os.path.splitext(data_path)[1].lower()
        
        if file_ext == '.csv':
            data = pd.read_csv(data_path)
        elif file_ext in ['.xls', '.xlsx']:
            data = pd.read_excel(data_path)
        elif file_ext == '.json':
            data = pd.read_json(data_path)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        return data
    
    def _preprocess_data(self, data):
        """
        Preprocess the data.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Preprocessed train, validation, and test sets
        """
        if self.target_column is None or self.target_column not in data.columns:
            raise ValueError(f"Target column not found: {self.target_column}")
        
        X = data.drop(columns=[self.target_column])
        y = data[self.target_column]
        
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)
        
        preprocessor = DataPreprocessor(
            text_columns=self.text_columns,
            numerical_columns=self.numerical_columns,
            categorical_columns=self.categorical_columns
        )
        
        X_train_processed = preprocessor.fit_transform(X_train)
        
        X_val_processed = preprocessor.transform(X_val)
        X_test_processed = preprocessor.transform(X_test)
        
        joblib.dump(preprocessor, os.path.join(self.output_dir, 'preprocessor.joblib'))
        
        return X_train_processed, X_val_processed, X_test_processed, y_train, y_val, y_test
    
    def _extract_nlp_features(self, X_train, X_val, X_test):
        """
        Extract NLP features from text columns.
        
        Args:
            X_train: Training features
            X_val: Validation features
            X_test: Test features
            
        Returns:
            Updated features with NLP features
        """
        nlp_extractor = NLPFeatureExtractor(method='tfidf')
        
        for col in self.text_columns:
            train_features = nlp_extractor.fit_transform(X_train[col])
            val_features = nlp_extractor.transform(X_val[col])
            test_features = nlp_extractor.transform(X_test[col])
            
            feature_names = [f"{col}_nlp_{i}" for i in range(train_features.shape[1])]
            train_nlp_df = pd.DataFrame(train_features.toarray(), columns=feature_names, index=X_train.index)
            val_nlp_df = pd.DataFrame(val_features.toarray(), columns=feature_names, index=X_val.index)
            test_nlp_df = pd.DataFrame(test_features.toarray(), columns=feature_names, index=X_test.index)
            
            X_train = pd.concat([X_train, train_nlp_df], axis=1)
            X_val = pd.concat([X_val, val_nlp_df], axis=1)
            X_test = pd.concat([X_test, test_nlp_df], axis=1)
            
            X_train = X_train.drop(columns=[col])
            X_val = X_val.drop(columns=[col])
            X_test = X_test.drop(columns=[col])
        
        return X_train, X_val, X_test
    
    def _train_traditional_model(self, X_train, X_val, y_train, y_val):
        """
        Train a traditional machine learning model.
        
        Args:
            X_train: Training features
            X_val: Validation features
            y_train: Training targets
            y_val: Validation targets
            
        Returns:
            Trained model and evaluation metrics
        """
        model_builder = ModelBuilder(model_type='rf', tune_hyperparams=True)
        
        model = model_builder.build_model()
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else None
        
        metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred, average='binary'),
            'recall': recall_score(y_val, y_pred, average='binary'),
            'f1': f1_score(y_val, y_pred, average='binary')
        }
        
        if y_prob is not None:
            metrics['roc_auc'] = roc_auc_score(y_val, y_prob)
        
        print("Validation Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        joblib.dump(model, os.path.join(self.output_dir, 'traditional_model.joblib'))
        
        return model, metrics
    
    def _train_deep_learning_model(self, X_train, X_val, y_train, y_val):
        """
        Train a deep learning model.
        
        Args:
            X_train: Training features
            X_val: Validation features
            y_train: Training targets
            y_val: Validation targets
            
        Returns:
            Trained model and evaluation metrics
        """
        dl_model = DeepLearningModel(model_type='cnn')
        
        model = dl_model.build_model(input_shape=(X_train.shape[1],))
        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=32,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
            ]
        )
        
        y_pred = (model.predict(X_val) > 0.5).astype(int)
        y_prob = model.predict(X_val)
        
        metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred, average='binary'),
            'recall': recall_score(y_val, y_pred, average='binary'),
            'f1': f1_score(y_val, y_pred, average='binary'),
            'roc_auc': roc_auc_score(y_val, y_prob)
        }
        
        print("Validation Metrics (Deep Learning):")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        model.save(os.path.join(self.output_dir, 'deep_learning_model'))
        
        return model, metrics
    
    def _evaluate_models(self, traditional_model, dl_model, X_test, y_test):
        """
        Evaluate models on test set.
        
        Args:
            traditional_model: Trained traditional model
            dl_model: Trained deep learning model
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary with evaluation results
        """
        results = {}
        
        if traditional_model is not None:
            y_pred = traditional_model.predict(X_test)
            y_prob = traditional_model.predict_proba(X_test)[:, 1] if hasattr(traditional_model, 'predict_proba') else None
            
            traditional_metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='binary'),
                'recall': recall_score(y_test, y_pred, average='binary'),
                'f1': f1_score(y_test, y_pred, average='binary')
            }
            
            if y_prob is not None:
                traditional_metrics['roc_auc'] = roc_auc_score(y_test, y_prob)
            
            cm = confusion_matrix(y_test, y_pred)
            
            print("Test Metrics (Traditional Model):")
            for metric, value in traditional_metrics.items():
                print(f"{metric}: {value:.4f}")
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix (Traditional Model)')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.savefig(os.path.join(self.output_dir, 'traditional_model_cm.png'))
            plt.close()
            
            results['traditional'] = {
                'metrics': traditional_metrics,
                'confusion_matrix': cm.tolist()
            }
        
        if dl_model is not None:
            y_pred = (dl_model.predict(X_test) > 0.5).astype(int)
            y_prob = dl_model.predict(X_test)
            
            dl_metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='binary'),
                'recall': recall_score(y_test, y_pred, average='binary'),
                'f1': f1_score(y_test, y_pred, average='binary'),
                'roc_auc': roc_auc_score(y_test, y_prob)
            }
            
            cm = confusion_matrix(y_test, y_pred)
            
            print("Test Metrics (Deep Learning Model):")
            for metric, value in dl_metrics.items():
                print(f"{metric}: {value:.4f}")
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix (Deep Learning Model)')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.savefig(os.path.join(self.output_dir, 'deep_learning_model_cm.png'))
            plt.close()
            
            results['deep_learning'] = {
                'metrics': dl_metrics,
                'confusion_matrix': cm.tolist()
            }
        
        return results
    
    def _save_results(self, traditional_model, dl_model, test_results):
        """
        Save results to files.
        
        Args:
            traditional_model: Trained traditional model
            dl_model: Trained deep learning model
            test_results: Dictionary with test results
        """
        import json
        
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj
        
        serializable_results = convert_to_serializable(test_results)
        
        with open(os.path.join(self.output_dir, 'test_results.json'), 'w') as f:
            json.dump(serializable_results, f, indent=4)
        
        if traditional_model is not None and hasattr(traditional_model, 'feature_importances_'):
            importances = traditional_model.feature_importances_
            
            if hasattr(traditional_model, 'feature_names_in_'):
                feature_names = traditional_model.feature_names_in_
            else:
                feature_names = [f'feature_{i}' for i in range(len(importances))]
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            importance_df.to_csv(os.path.join(self.output_dir, 'feature_importance.csv'), index=False)
            
            plt.figure(figsize=(12, 8))
            sns.barplot(x='importance', y='feature', data=importance_df.head(20))
            plt.title('Top 20 Feature Importances')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'feature_importance.png'))
            plt.close()
        
        if 'traditional' in test_results and 'deep_learning' in test_results:
            traditional_metrics = test_results['traditional']['metrics']
            dl_metrics = test_results['deep_learning']['metrics']
            
            metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
            comparison_data = []
            
            for metric in metrics:
                if metric in traditional_metrics and metric in dl_metrics:
                    comparison_data.append({
                        'metric': metric,
                        'traditional': traditional_metrics[metric],
                        'deep_learning': dl_metrics[metric]
                    })
            
            comparison_df = pd.DataFrame(comparison_data)
            
            comparison_df.to_csv(os.path.join(self.output_dir, 'model_comparison.csv'), index=False)
            
            plt.figure(figsize=(10, 6))
            comparison_df.set_index('metric').plot(kind='bar')
            plt.title('Model Comparison')
            plt.ylabel('Score')
            plt.ylim(0, 1)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'model_comparison.png'))
            plt.close()
        
        with open(os.path.join(self.output_dir, 'summary_report.txt'), 'w') as f:
            f.write("Heart Disease Detection - Summary Report\n")
            f.write("======================================\n\n")
            
            f.write("Dataset Information:\n")
            f.write("-----------------\n")
            f.write(f"Target Column: {self.target_column}\n")
            f.write(f"Numerical Columns: {', '.join(self.numerical_columns)}\n")
            f.write(f"Categorical Columns: {', '.join(self.categorical_columns)}\n")
            f.write(f"Text Columns: {', '.join(self.text_columns)}\n\n")
            
            if 'traditional' in test_results:
                f.write("Traditional Model Results:\n")
                f.write("------------------------\n")
                for metric, value in test_results['traditional']['metrics'].items():
                    f.write(f"{metric}: {value:.4f}\n")
                f.write("\n")
            
            if 'deep_learning' in test_results:
                f.write("Deep Learning Model Results:\n")
                f.write("---------------------------\n")
                for metric, value in test_results['deep_learning']['metrics'].items():
                    f.write(f"{metric}: {value:.4f}\n")
                f.write("\n")
            
            f.write("Conclusion:\n")
            f.write("----------\n")
            
            if 'traditional' in test_results and 'deep_learning' in test_results:
                traditional_f1 = test_results['traditional']['metrics'].get('f1', 0)
                dl_f1 = test_results['deep_learning']['metrics'].get('f1', 0)
                
                if traditional_f1 > dl_f1:
                    f.write("The traditional model performs better based on F1 score.\n")
                    best_model = "traditional"
                else:
                    f.write("The deep learning model performs better based on F1 score.\n")
                    best_model = "deep_learning"
                
                f.write("\nDetailed Analysis:\n")
                
                traditional_precision = test_results['traditional']['metrics'].get('precision', 0)
                traditional_recall = test_results['traditional']['metrics'].get('recall', 0)
                dl_precision = test_results['deep_learning']['metrics'].get('precision', 0)
                dl_recall = test_results['deep_learning']['metrics'].get('recall', 0)
                
                f.write(f"Traditional Model - Precision: {traditional_precision:.4f}, Recall: {traditional_recall:.4f}\n")
                f.write(f"Deep Learning Model - Precision: {dl_precision:.4f}, Recall: {dl_recall:.4f}\n\n")
                
                if traditional_precision > dl_precision and traditional_recall < dl_recall:
                    f.write("The traditional model has higher precision but lower recall than the deep learning model.\n")
                    f.write("This means it makes fewer false positive errors but more false negative errors.\n")
                elif traditional_precision < dl_precision and traditional_recall > dl_recall:
                    f.write("The traditional model has lower precision but higher recall than the deep learning model.\n")
                    f.write("This means it makes more false positive errors but fewer false negative errors.\n")
                
                f.write("\nRecommendation:\n")
                if best_model == "traditional":
                    f.write("Use the traditional model for production as it shows better overall performance.\n")
                else:
                    f.write("Use the deep learning model for production as it shows better overall performance.\n")
            
            elif 'traditional' in test_results:
                f.write("Only the traditional model was evaluated. Its performance is satisfactory for heart disease detection.\n")
            elif 'deep_learning' in test_results:
                f.write("Only the deep learning model was evaluated. Its performance is satisfactory for heart disease detection.\n")
            else:
                f.write("No models were evaluated. Please check the pipeline configuration and try again.\n")
            from datetime import datetime
            f.write(f"\nReport generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
        print(f"Results saved to {self.output_dir}")

