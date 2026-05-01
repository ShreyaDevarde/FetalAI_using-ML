from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# ✅ Load model + scaler
model, scaler = pickle.load(open("fetal_health_final.pkl", "rb"))

# ✅ Feature names (MUST match training)
FEATURE_NAMES = [
    'prolongued_decelerations',
    'abnormal_short_term_variability',
    'percentage_of_time_with_abnormal_long_term_variability',
    'histogram_variance',
    'histogram_median',
    'mean_value_of_long_term_variability',
    'histogram_mode',
    'accelerations'
]

# Label mapping
CLASS_LABELS = {
    1: "Normal",
    2: "Suspect",
    3: "Pathological"
}

# Styling + description
CLASS_CSS = {
    "Normal": "normal",
    "Suspect": "suspect",
    "Pathological": "pathological"
}

CLASS_DESCRIPTIONS = {
    "Normal": "Healthy fetal condition. No signs of distress.",
    "Suspect": "Requires monitoring and clinical evaluation.",
    "Pathological": "Immediate medical attention required."
}


# ---------------------------
# HOME
# ---------------------------
@app.route('/')
def home():
    return render_template('index.html')


# ---------------------------
# PREDICTION
# ---------------------------
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('inspect.html')

    try:
        # ✅ Step 1: Collect input
        input_dict = {
            'prolongued_decelerations': float(request.form['prolongued_decelerations']),
            'abnormal_short_term_variability': float(request.form['abnormal_short_term_variability']),
            'percentage_of_time_with_abnormal_long_term_variability': float(request.form['percentage_of_time_with_abnormal_long_term_variability']),
            'histogram_variance': float(request.form['histogram_variance']),
            'histogram_median': float(request.form['histogram_median']),
            'mean_value_of_long_term_variability': float(request.form['mean_value_of_long_term_variability']),
            'histogram_mode': float(request.form['histogram_mode']),
            'accelerations': float(request.form['accelerations'])
        }

        # ✅ Step 2: Convert to DataFrame (IMPORTANT FIX)
        input_df = pd.DataFrame([input_dict], columns=FEATURE_NAMES)

        # ✅ Step 3: Apply scaling (same as training)
        input_scaled = scaler.transform(input_df)

        # ✅ Step 4: Predict
        prediction = model.predict(input_scaled)[0]

        # ✅ Step 5: Map result
        result = CLASS_LABELS.get(int(prediction), "Unknown")

        # UI helpers
        result_class = CLASS_CSS.get(result, 'normal')
        description = CLASS_DESCRIPTIONS.get(result, '')
        proba = model.predict_proba(input_scaled)
        confidence = round(max(proba[0]) * 100, 2)

        return render_template(
            'output.html',
            prediction=result,
            confidence=confidence,
            result_class=result_class,
            description=description,
            features=input_dict
        )

    except Exception as e:
        return f"Error: {e}"


# ---------------------------
# RUN
# ---------------------------
if __name__ == "__main__":
    app.run(debug=True)