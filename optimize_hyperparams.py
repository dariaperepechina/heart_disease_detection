import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score, make_scorer

print("Download data...")
train_df = pd.read_csv('data/split/train.csv')
val_df = pd.read_csv('data/split/val.csv')

target_col = train_df.columns[-1]
X_train = train_df.drop(columns=[target_col])
y_train = train_df[target_col]
X_val = val_df.drop(columns=[target_col])
y_val = val_df[target_col]

encoder = joblib.load('models/label_encoder.pkl')
y_train = encoder.transform(y_train)
y_val = encoder.transform(y_val)

param_dist = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False],
    'class_weight': ['balanced', 'balanced_subsample', None]
}

rf = RandomForestClassifier(random_state=42)

f1_scorer = make_scorer(f1_score, average='weighted')

print("Searching of hyperparameters")
search = RandomizedSearchCV(
    rf, 
    param_distributions=param_dist,
    n_iter=20, 
    scoring=f1_scorer,
    cv=5,
    verbose=1,
    n_jobs=-1,
    random_state=42
)

search.fit(X_train, y_train)

print(f"Best parameters: {search.best_params_}")
print(f"Best F1-score: {search.best_score_:.4f}")

best_model = search.best_estimator_
y_pred = best_model.predict(X_val)
val_f1 = f1_score(y_val, y_pred, average='weighted')
print(f"F1-score on validation: {val_f1:.4f}")

joblib.dump(best_model, 'models/rf_optimized_model.pkl')
print("models/rf_optimized_model.pkl")

results = pd.DataFrame(search.cv_results_)
results.to_csv('models/hyperparameter_search_results.csv', index=False)
print("models/hyperparameter_search_results.csv")
