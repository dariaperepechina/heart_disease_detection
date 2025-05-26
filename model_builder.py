from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

class ModelBuilder:
    def __init__(self, model_type='rf', tune_hyperparams=False):
        """
        Initialize the model builder.
        
        Args:
            model_type: Type of model to build ('rf', 'gb', 'lr', or 'svm')
            tune_hyperparams: Whether to tune hyperparameters
        """
        self.model_type = model_type
        self.tune_hyperparams = tune_hyperparams
    
    def build_model(self):
        """
        Build and return a model.
        
        Returns:
            Scikit-learn model
        """
        if self.model_type == 'rf':
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42
            )
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        elif self.model_type == 'gb':
            model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            )
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
        elif self.model_type == 'lr':
            model = LogisticRegression(
                C=1.0,
                penalty='l2',
                solver='liblinear',
                random_state=42
            )
            param_grid = {
                'C': [0.1, 1.0, 10.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            }
        elif self.model_type == 'svm':
            model = SVC(
                C=1.0,
                kernel='rbf',
                probability=True,
                random_state=42
            )
            param_grid = {
                'C': [0.1, 1.0, 10.0],
                'kernel': ['linear', 'rbf', 'poly'],
                'gamma': ['scale', 'auto', 0.1, 0.01]
            }
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        if self.tune_hyperparams:
            print(f"Tuning hyperparameters for {self.model_type}...")
            model = GridSearchCV(
                model,
                param_grid,
                cv=5,
                scoring='f1',
                n_jobs=-1
            )
        
        return model
