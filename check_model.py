import joblib
import pandas as pd

model = joblib.load('models/rf_optimized_model.pkl')

print("Model features:")
for i, feature in enumerate(model.feature_names_in_):
    print(f"{i+1}. {feature}")

high_risk_data = {feature: 0 for feature in model.feature_names_in_}

high_risk_data['age'] = 75
high_risk_data['chest_pain'] = 1
high_risk_data['systolic.blood.pressure'] = 180

test_df = pd.DataFrame([high_risk_data])

prediction = model.predict(test_df)[0]
probability = model.predict_proba(test_df)[0][1]

print(f"\nTest prediction: {prediction}")
print(f"Probability: {probability:.4f}")
