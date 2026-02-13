from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier

app = Flask(__name__)

# --- 1. ML ENGINE CONFIGURATION ---
def initialize_engine():
    file_path = os.path.join(os.path.dirname(__file__), 'diabetes_dataset.csv')
    if not os.path.exists(file_path):
        return None, None, None

    df = pd.read_csv(file_path)
    data = df.copy()
    le = LabelEncoder()
    data['gender'] = le.fit_transform(data['gender'])
    X = data.drop(['diabetes', 'location', 'year'], axis=1, errors='ignore')
    if 'smoking_history' in X.columns:
        X = pd.get_dummies(X, columns=['smoking_history'], drop_first=True)
    y = data['diabetes']
    feature_names = X.columns.tolist()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = HistGradientBoostingClassifier(random_state=42, class_weight='balanced')
    model.fit(X_scaled, y)
    return model, scaler, feature_names

MODEL, SCALER, FEATURES = initialize_engine()

# --- 2. ROUTING LOGIC ---
@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    # Requirement: Capture the form data to send it back to the UI
    form_data = request.form if request.method == 'POST' else {}
    
    if request.method == 'POST':
        try:
            # Note: request.form.get('name') must match HTML <input name="...">
            gender_val = 1 if request.form.get('gender') == 'Male' else 0
            age = float(request.form.get('age', 0))
            hypertension = int(request.form.get('hypertension', 0))
            heart_disease = int(request.form.get('heart_disease', 0))
            bmi = float(request.form.get('bmi', 0))
            hba1c = float(request.form.get('hba1c', 0))
            glucose = float(request.form.get('glucose', 0))

            input_df = pd.DataFrame(0.0, index=[0], columns=FEATURES)
            mappings = {
                'gender': gender_val,
                'age': age,
                'hypertension': hypertension,
                'heart_disease': heart_disease,
                'bmi': bmi,
                'hbA1c_level': hba1c,
                'blood_glucose_level': glucose
            }
            
            for col, val in mappings.items():
                if col in FEATURES:
                    input_df.loc[0, col] = val

            scaled_input = SCALER.transform(input_df)
            probability = MODEL.predict_proba(scaled_input)[0][1]

            if probability > 0.70:
                cat, css, rec = "High Risk", "status-high", "Immediate clinical consultation and diagnostic testing required."
            elif probability > 0.30:
                cat, css, rec = "Moderate Risk", "status-moderate", "Routine monitoring advised. Schedule follow-up test within 6 months."
            else:
                cat, css, rec = "Low Risk", "status-low", "No immediate intervention required."

            result = {
                "probability": f"{probability:.1%}",
                "category": cat,
                "css_class": css,
                "recommendation": rec,
                "data_summary": {
                    "Age": f"{age} YOA",
                    "HbA1c": f"{hba1c}%",
                    "BMI": f"{bmi}"
                }
            }
        except Exception as e:
            result = {"error": str(e)}

    # Pass result AND form_data back to the HTML
    return render_template('index.html', result=result, form_data=form_data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
