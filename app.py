from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle

app = Flask(__name__)

# Load the trained model
model = load_model('heart_disease_model.h5')

# Function to preprocess input
def preprocess_input(form_data):
    sex = 1 if form_data['Sex'] == 'M' else 0
    chest_pain = {'TA': 0, 'ATA': 1, 'NAP': 2, 'ASY': 3}[form_data['ChestPainType']]
    resting_ecg = {'Normal': 0, 'ST': 1, 'LVH': 2}[form_data['RestingECG']]
    exercise_angina = 1 if form_data['ExerciseAngina'] == 'Y' else 0
    st_slope = {'Up': 0, 'Flat': 1, 'Down': 2}[form_data['ST_Slope']]

    values = [
        float(form_data['Age']),
        sex,
        float(form_data['RestingBP']),
        float(form_data['Cholesterol']),
        int(form_data['FastingBS']),
        chest_pain,
        resting_ecg,
        float(form_data['MaxHR']),
        exercise_angina,
        float(form_data['Oldpeak']),
        st_slope
    ]
    return np.array([values])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        data = request.form
        input_data = preprocess_input(data)

        prediction = model.predict(input_data)
        predicted_class = (prediction[0][0] > 0.5)  # Assuming sigmoid output

        if predicted_class:
            result = "The person is likely to have heart disease."
        else:
            result = "The person is not likely to have heart disease."

        return render_template('index.html', prediction_text=result)

if __name__ == '__main__':
    app.run(debug=True)
