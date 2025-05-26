import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten, LSTM, Bidirectional, Input, Embedding, concatenate
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class DeepLearningModel:
    def __init__(self, model_type='cnn', max_words=10000, max_sequence_length=500, embedding_dim=100):
        """
        Initialize the deep learning model.
        
        Args:
            model_type: Type of model to build ('cnn', 'lstm', 'bilstm', or 'hybrid')
            max_words: Maximum number of words in the vocabulary
            max_sequence_length: Maximum sequence length
            embedding_dim: Dimension of word embeddings
        """
        self.model_type = model_type
        self.max_words = max_words
        self.max_sequence_length = max_sequence_length
        self.embedding_dim = embedding_dim
        self.model = None
        self.tokenizer = None
    
    def build_model(self, input_shape):
        """
        Build and compile the model.
        
        Args:
            input_shape: Shape of the input data
            
        Returns:
            Compiled Keras model
        """
        if self.model_type == 'cnn':
            model = Sequential([
                Dense(128, activation='relu', input_shape=(input_shape,)),
                Dropout(0.3),
                Dense(64, activation='relu'),
                Dropout(0.3),
                Dense(1, activation='sigmoid')
            ])
        elif self.model_type == 'lstm':
            model = Sequential([
                Dense(128, activation='relu', input_shape=(input_shape,)),
                Dropout(0.3),
                Dense(64, activation='relu'),
                Dropout(0.3),
                Dense(1, activation='sigmoid')
            ])
        elif self.model_type == 'bilstm':
            model = Sequential([
                Dense(128, activation='relu', input_shape=(input_shape,)),
                Dropout(0.3),
                Dense(64, activation='relu'),
                Dropout(0.3),
                Dense(1, activation='sigmoid')
            ])
        elif self.model_type == 'hybrid':
            model = Sequential([
                Dense(128, activation='relu', input_shape=(input_shape,)),
                Dropout(0.3),
                Dense(64, activation='relu'),
                Dropout(0.3),
                Dense(1, activation='sigmoid')
            ])
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def fit(self, X_train, y_train, X_val, y_val, epochs=20, batch_size=32):
        """
        Fit the model.
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            epochs: Number of epochs
            batch_size: Batch size
            
        Returns:
            Training history
        """
        input_shape = X_train.shape[1]
        self.model = self.build_model(input_shape)
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping]
        )
        
        return history
    
    def predict(self, X):
        """
        Make predictions.
        
        Args:
            X: Input data
            
        Returns:
            Predicted probabilities
        """
        if self.model is None:
            raise ValueError("Model has not been trained. Call fit first.")
        
        return self.model.predict(X).flatten()
    
    def save(self, path):
        """
        Save the model.
        
        Args:
            path: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model has not been trained. Call fit first.")
        
        self.model.save(path)
