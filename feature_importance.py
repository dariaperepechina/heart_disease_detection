import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os

os.makedirs('plots', exist_ok=True)

print("Downloadind model and data...")
model = joblib.load('models/rf_optimized_model.pkl')
train_df = pd.read_csv('data/split/train.csv')

target_col = train_df.columns[-1]
feature_names = train_df.drop(columns=[target_col]).columns

importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

print("Feature importance:")
for i in range(min(20, len(feature_names))):
    print(f"{i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
})
importance_df = importance_df.sort_values('Importance', ascending=False)
importance_df.to_csv('plots/feature_importance.csv', index=False)
print("plots/feature_importance.csv")

plt.figure(figsize=(12, 8))
plt.title("Feature importance Top 20")
plt.bar(range(min(20, len(feature_names))), 
        importances[indices[:20]],
        align="center")
plt.xticks(range(min(20, len(feature_names))), 
           [feature_names[i] for i in indices[:20]], 
           rotation=90)
plt.tight_layout()
plt.savefig('plots/feature_importance_top20.png')
print("plots/feature_importance_top20.png")

from sklearn.metrics import roc_curve, auc

test_df = pd.read_csv('data/split/test.csv')
X_test = test_df.drop(columns=[target_col])
y_test = test_df[target_col]

encoder = joblib.load('models/label_encoder.pkl')
y_test = encoder.transform(y_test)

y_proba = model.predict_proba(X_test)[:, 1]

fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, 
         label=f'ROC (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('plots/roc_curve.png')
print("plots/roc_curve.png")
