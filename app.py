from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the saved model
# Make sure your trained model is saved as 'fetal_healthAI.pkl' in the same directory
model, scaler = pickle.load(open('fetal_health.pkl', 'rb'))

# Map model output to label
CLASS_LABELS = {
    1: 'Normal',
    2: 'Suspect',
    3: 'Pathological'
}

# CSS class per result (used to style outputt.html)
CLASS_CSS = {
    'Normal':        'normal',
    'Suspect':       'suspect',
    'Pathological':  'pathological'
}

# Human-readable description per result
CLASS_DESCRIPTIONS = {
    'Normal': (
        "The fetal CTG data indicates a healthy state with no signs of "
        "hypoxia or acidosis. Continue routine monitoring as advised by your clinician."
    ),
    'Suspect': (
        "The CTG pattern requires clinical evaluation. There are features that "
        "may need further monitoring or assessment. Please consult your obstetrician promptly."
    ),
    'Pathological': (
        "The CTG pattern suggests a potentially dangerous fetal state. "
        "Immediate medical evaluation and possible intervention is strongly recommended."
    )
}


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('inspect.html')

    try:
        # ✅ Correct feature order (same as training)
        input_data = [
            float(request.form['prolongued_decelerations']),
            float(request.form['abnormal_short_term_variability']),
            float(request.form['percentage_of_time_with_abnormal_long_term_variability']),
            float(request.form['histogram_variance']),
            float(request.form['histogram_median']),
            float(request.form['mean_value_of_long_term_variability']),
            float(request.form['histogram_mode']),
            float(request.form['accelerations'])
        ]
    except:
        return "Invalid input. Please enter valid numbers."

    # ✅ Create dictionary for display (fixes Jinja error)
    features = {
        "prolongued_decelerations": input_data[0],
        "abnormal_short_term_variability": input_data[1],
        "percentage_of_time_with_abnormal_long_term_variability": input_data[2],
        "histogram_variance": input_data[3],
        "histogram_median": input_data[4],
        "mean_value_of_long_term_variability": input_data[5],
        "histogram_mode": input_data[6],
        "accelerations": input_data[7]
    }

    # ✅ Convert to array
    input_array = np.array(input_data).reshape(1, -1)

    # ✅ Apply scaling
    input_array = scaler.transform(input_array)

    # ✅ Prediction
    prediction = model.predict(input_array)[0]

    # ✅ Map result
    result = CLASS_LABELS.get(int(prediction), "Unknown")

    # Optional styling
    result_class = CLASS_CSS.get(result, 'normal')
    description = CLASS_DESCRIPTIONS.get(result, '')

    return render_template(
        'output.html',
        prediction=result,
        result_class=result_class,
        description=description,
        features=features
    )

if __name__ == '__main__':
    app.run(debug=True)