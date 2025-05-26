import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class NLPFeatureExtractor:
    def __init__(self, method='tfidf', max_features=5000):
        """
        Initialize the NLP feature extractor.
        
        Args:
            method: Feature extraction method ('tfidf', 'bow', 'bert', or 'spacy')
            max_features: Maximum number of features to extract
        """
        self.method = method
        self.max_features = max_features
        self.vectorizer = None
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
    
    def preprocess_text(self, text):
        """
        Preprocess text data.
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        text = text.lower()
        
        tokens = word_tokenize(text)
        
        tokens = [self.stemmer.stem(token) for token in tokens if token.isalnum() and token not in self.stop_words]
        
        return ' '.join(tokens)
    
    def fit_transform(self, texts):
        """
        Fit the vectorizer and transform the texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            Feature matrix
        """
        preprocessed_texts = [self.preprocess_text(text) for text in texts]
        
        if self.method == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                min_df=2,
                max_df=0.95
            )
        elif self.method == 'bow':
            self.vectorizer = CountVectorizer(
                max_features=self.max_features,
                min_df=2,
                max_df=0.95
            )
        elif self.method == 'bert':
            print("BERT embeddings require additional dependencies. Using TF-IDF instead.")
            self.vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                min_df=2,
                max_df=0.95
            )
        elif self.method == 'spacy':
            print("SpaCy embeddings require additional dependencies. Using TF-IDF instead.")
            self.vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                min_df=2,
                max_df=0.95
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        features = self.vectorizer.fit_transform(preprocessed_texts)
        
        print(f"NLP features extracted: {features.shape[1]} features")
        
        return features
    
    def transform(self, texts):
        """
        Transform texts using the fitted vectorizer.
        
        Args:
            texts: List of text strings
            
        Returns:
            Feature matrix
        """
        if self.vectorizer is None:
            raise ValueError("Vectorizer has not been fitted. Call fit_transform first.")
        
        preprocessed_texts = [self.preprocess_text(text) for text in texts]
        
        features = self.vectorizer.transform(preprocessed_texts)
        
        return features
