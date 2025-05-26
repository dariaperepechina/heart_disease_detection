import joblib
import pandas as pd
import numpy as np

model = joblib.load('models/rf_optimized_model.pkl')

base_case = {feature: 0 for feature in model.feature_names_in_}

test_cases = []

case1 = base_case.copy()
case1['ageCat_(79,89]'] = 1
case1['systolic.blood.pressure'] = 190
case1['diastolic.blood.pressure'] = 110
case1['high.sensitivity.troponin'] = 2.0
case1['pulse'] = 120
test_cases.append(("Elderly with high BP", case1))

case2 = base_case.copy()
case2['ageCat_(59,69]'] = 1
case2['diabetes'] = 1
case2['congestive.heart.failure'] = 1
case2['cholesterol'] = 300
test_cases.append(("Middle-aged with conditions", case2))

case3 = base_case.copy()
case3['high.sensitivity.troponin'] = 5.0
case3['NYHA.cardiac.function.classification_IV'] = 1
case3['Killip.grade_IV'] = 1
test_cases.append(("Critical cardiac values", case3))

case4 = base_case.copy()
case4['high.sensitivity.troponin'] = 10.0
case4['myoglobin'] = 1000
case4['creatine.kinase'] = 1000
test_cases.append(("Extreme lab values", case4))

print("Testing combinations for high risk predictions:\n")
for name, case in test_cases:
    df = pd.DataFrame([case])
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]
    result = "HIGH RISK" if prediction == 1 else "Low risk"
    print(f"{name}: {result} (Probability: {probability:.2f})")
    
    if prediction == 1:
        print("  Key values that triggered high risk:")
        for key, value in case.items():
            if value > 0:
                print(f"  - {key}: {value}")
        print()
