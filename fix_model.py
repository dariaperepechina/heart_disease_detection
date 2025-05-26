import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

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

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',  
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_val)
print(classification_report(y_val, y_pred))

joblib.dump(model, 'models/random_forest_balanced_model.pkl')
print("models/random_forest_balanced_model.pkl")
