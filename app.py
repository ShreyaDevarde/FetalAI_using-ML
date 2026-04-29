from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the saved model
# Make sure your trained model is saved as 'fetal_healthAI.pkl' in the same directory
model = pickle.load(open('fetal_healthAI.pkl', 'rb'))

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

    # Collect feature values from the form
    feature_names = [
        #'baseline value',
        'accelerations',
        #'fetal_movement',
        #'uterine_contractions',
        #'light_decelerations',
        #'severe_decelerations',
        'prolongued_decelerations',
        'abnormal_short_term_variability',
        #'mean_value_of_short_term_variability',
        'percentage_of_time_with_abnormal_long_term_variability',
        'mean_value_of_long_term_variability',
        #'histogram_width',
        #'histogram_min',
        #'histogram_max',
        #'histogram_number_of_peaks',
        #'histogram_number_of_zeroes',
        'histogram_mode',
        #'histogram_mean',
        'histogram_median',
        'histogram_variance',
    ]

    features = {}
    values = []
    for name in feature_names:
        val = request.form.get(name, 0)
        try:
            val = float(val)
        except ValueError:
            val = 0.0
        features[name] = val
        values.append(val)

    # Run prediction
    input_array = np.array([values])
    raw_pred = model.predict(input_array)[0]

    # raw_pred may be 1/2/3 (int) or the label string depending on your model
    if isinstance(raw_pred, (int, float, np.integer)):
        prediction = CLASS_LABELS.get(int(raw_pred), str(raw_pred))
    else:
        prediction = str(raw_pred)

    result_class  = CLASS_CSS.get(prediction, 'normal')
    description   = CLASS_DESCRIPTIONS.get(prediction, '')

    return render_template(
        'output.html',
        prediction=prediction,
        result_class=result_class,
        description=description,
        features=features,
    )


if __name__ == '__main__':
    app.run(debug=True)