import argparse
import os
import json
from pipeline import Pipeline
from data_loader import DataLoader

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Heart Disease Detection Pipeline')
    
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory containing the dataset files')
    
    parser.add_argument('--config_path', type=str, default=None,
                        help='Path to configuration JSON file')
    
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save results')
    
    parser.add_argument('--auto_detect_columns', action='store_true',
                        help='Automatically detect column types')
    
    parser.add_argument('--text_columns', type=str, nargs='+',
                        help='Names of text columns containing symptom descriptions')
    
    parser.add_argument('--numerical_columns', type=str, nargs='+',
                        help='Names of numerical feature columns')
    
    parser.add_argument('--categorical_columns', type=str, nargs='+',
                        help='Names of categorical feature columns')
    
    parser.add_argument('--target_column', type=str,
                        help='Name of the target column')
    
    parser.add_argument('--model_type', type=str, default='rf',
                        choices=['rf', 'gb', 'lr', 'svm'],
                        help='Type of traditional ML model to use')
    
    parser.add_argument('--dl_model_type', type=str, default='cnn',
                        choices=['cnn', 'lstm', 'bilstm', 'hybrid'],
                        help='Type of deep learning model to use')
    
    parser.add_argument('--tune_hyperparams', action='store_true',
                        help='Whether to tune hyperparameters')
    
    parser.add_argument('--nlp_method', type=str, default='tfidf',
                        choices=['tfidf', 'bow', 'bert', 'spacy'],
                        help='NLP feature extraction method')
    
    return parser.parse_args()

def main():
    """Run the heart disease detection pipeline."""
    args = parse_args()
    
    if args.config_path and os.path.exists(args.config_path):
        with open(args.config_path, 'r') as f:
            config = json.load(f)
    else:
        config = {
            'output_dir': args.output_dir,
            'model_type': args.model_type,
            'dl_model_type': args.dl_model_type,
            'tune_hyperparams': args.tune_hyperparams,
            'nlp_method': args.nlp_method,
            'max_features': 5000,
            'max_words': 10000,
            'max_sequence_length': 500,
            'embedding_dim': 100,
            'epochs': 20,
            'batch_size': 32
        }
    
    loader = DataLoader(args.data_dir)
    df = loader.load_data()
    
    if args.auto_detect_columns:
        column_types = loader.identify_column_types(df)
        config.update(column_types)
        print("\nAutomatically detected column types:")
        for type_name, columns in column_types.items():
            print(f"\n{type_name}:")
            for col in columns:
                print(f"  - {col}")
    else:
        if args.text_columns:
            config['text_columns'] = args.text_columns
        if args.numerical_columns:
            config['numerical_columns'] = args.numerical_columns
        if args.categorical_columns:
            config['categorical_columns'] = args.categorical_columns
        if args.target_column:
            config['target_column'] = args.target_column
    
    os.makedirs(config['output_dir'], exist_ok=True)
    
    config_path = os.path.join(config['output_dir'], 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"Configuration saved to {config_path}")
    
    pipeline = Pipeline(config)
    results = pipeline.run(df)
    
    results_path = os.path.join(config['output_dir'], 'results.json')
    
    serializable_results = {}
    for model_name, result in results.items():
        serializable_results[model_name] = {
            'metrics': {k: float(v) for k, v in result['metrics'].items()}
        }
    
    with open(results_path, 'w') as f:
        json.dump(serializable_results, f, indent=4)
    
    print(f"Results saved to {results_path}")
    
    print("\nResults Summary:")
    for model_name, result in results.items():
        print(f"\n{model_name}:")
        for metric_name, metric_value in result['metrics'].items():
            print(f"  {metric_name}: {metric_value:.4f}")

if __name__ == "__main__":
    main()
