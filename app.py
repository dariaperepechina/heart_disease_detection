from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import re

app = Flask(__name__)

model = joblib.load('models/rf_optimized_model.pkl')

def analyze_symptoms(text):
    text = text.lower()
    
    heart_attack_symptoms = {
        'chest_pain': ['chest pain', 'chest discomfort', 'chest pressure', 'chest tightness'],
        'shortness_breath': ['shortness of breath', 'difficulty breathing', 'breathless'],
        'arm_pain': ['arm pain', 'shoulder pain', 'left arm', 'radiating pain'],
        'sweating': ['cold sweat', 'sweating', 'perspiration'],
        'nausea': ['nausea', 'vomiting', 'sick to stomach'],
        'fatigue': ['fatigue', 'tired', 'exhaustion', 'weakness'],
        'dizziness': ['dizzy', 'dizziness', 'lightheaded', 'faint'],
        'jaw_pain': ['jaw pain', 'jaw discomfort', 'throat pain']
    }
    
    detected_symptoms = {}
    risk_score = 0
    
    for symptom, keywords in heart_attack_symptoms.items():
        for keyword in keywords:
            if keyword in text:
                detected_symptoms[symptom] = 1
                risk_score += 1
                break
    
    return detected_symptoms, risk_score

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        symptom_text = data.pop('symptom_text', '')
        
        detected_symptoms, risk_score = analyze_symptoms(symptom_text)
        
        input_data = {feature: 0 for feature in model.feature_names_in_}
        
        for key, value in data.items():
            if key in input_data:
                input_data[key] = float(value)
        
        for symptom, value in detected_symptoms.items():
            if symptom in input_data:
                input_data[symptom] = value
        
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]
        
        adjusted_probability = min(probability + (risk_score * 0.05), 1.0)
        
        return jsonify({
            'prediction': int(prediction),
            'probability': float(probability),
            'adjusted_probability': float(adjusted_probability),
            'detected_symptoms': detected_symptoms,
            'message': 'High risk of myocardial infarction' if prediction == 1 else 'Low risk of myocardial infarction',
            'symptom_analysis': f"Detected {len(detected_symptoms)} symptoms related to heart attack."
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
