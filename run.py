import argparse
import os
from explore_dataset import DataExplorer
from preprocess import DataPreprocessor
from split_data import DataSplitter
from model import ModelTrainer
from evaluate import ModelEvaluator
from predict import Predictor

def main():
    parser = argparse.ArgumentParser(description='Heart Disease Detection Pipeline')
    
    parser.add_argument('--mode', type=str, required=True, 
                        choices=['explore', 'preprocess', 'split', 'train', 'evaluate', 'predict'],
                        help='Mode of operation')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the data file')
    
    parser.add_argument('--output_path', type=str, help='Path to save the output file')
    parser.add_argument('--output_dir', type=str, help='Directory to save output files')
    
    parser.add_argument('--train_path', type=str, help='Path to save the training data')
    parser.add_argument('--test_path', type=str, help='Path to save the test data')
    parser.add_argument('--val_path', type=str, help='Path to save the validation data')
    parser.add_argument('--test_size', type=float, default=0.2, help='Proportion of data to use for testing')
    parser.add_argument('--val_size', type=float, default=0.1, help='Proportion of data to use for validation')
    
    parser.add_argument('--model_type', type=str, default='logistic', 
                        choices=['logistic', 'random_forest'],
                        help='Type of model to train')
    parser.add_argument('--model_path', type=str, help='Path to save or load the model')
    
    args = parser.parse_args()
    
    if args.mode == 'explore':
        output_dir = args.output_dir or 'data_exploration'
        explorer = DataExplorer(output_dir=output_dir)
        explorer.explore(args.data_path)
    
    elif args.mode == 'preprocess':
        if not args.output_path:
            raise ValueError("Output path is required for preprocess mode")
        
        output_dir = args.output_dir or 'preprocessed'
        preprocessor = DataPreprocessor(output_dir=output_dir)
        preprocessor.preprocess(args.data_path, args.output_path)
    
    elif args.mode == 'split':
        if not args.train_path or not args.test_path:
            raise ValueError("Train and test paths are required for split mode")
        
        splitter = DataSplitter()
        splitter.split(
            args.data_path, 
            args.train_path, 
            args.test_path, 
            args.val_path, 
            args.test_size, 
            args.val_size
        )
    
    elif args.mode == 'train':
        if not args.data_path:
            raise ValueError("Data path is required for train mode")
        
        output_dir = args.output_dir or 'models'
        trainer = ModelTrainer(model_dir=output_dir)
        trainer.train(args.data_path, args.val_path, args.model_type)
    
    elif args.mode == 'evaluate':
        if not args.data_path or not args.model_path:
            raise ValueError("Data path and model path are required for evaluate mode")

        output_dir = args.output_dir or 'evaluation'
        evaluator = ModelEvaluator(output_dir=output_dir)
        evaluator.evaluate(args.model_path, args.data_path, args.output_path) 
    
    elif args.mode == 'predict':
        if not args.data_path or not args.model_path:
            raise ValueError("Data path and model path are required for predict mode")

        output_path = args.output_path or 'predictions.csv'
        predictor = Predictor(args.model_path, args.output_dir)  
        predictor.predict(args.data_path, output_path)
        

if __name__ == '__main__':
    main()
