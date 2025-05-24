
import numpy as np
from flask import Flask, request, render_template
import joblib


# Use joblib to load the model

app = Flask(__name__)

# Load your trained model

model = joblib.load(open("BMI_CLASS_PREDICTION.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    # Get form values
    age = float(request.form['Age'])
    height = float(request.form['Height'])
    weight = float(request.form['Weight'])

    # Auto-calculate BMI
    bmi = weight / (height ** 2)

    # Combine features for prediction
    final_features = [np.array([age, height, weight, bmi])]

    # Predict
    prediction = model.predict(final_features)

    # Map prediction to label
    if prediction[0] == 0:
        output = 'Normal Weight'
    elif prediction[0] == 1:
        output = 'Overweight'
    elif prediction[0] == 2:
        output = 'Underweight'
    elif prediction[0] == 3:
        output = 'Obese Class 3 (Very high level of obesity)'
    elif prediction[0] == 4:
        output = 'Obese Class 2 (More serious obesity)'
    else:
        output = 'Obese Class 1 (First level of obesity)'

    return render_template('index.html', prediction_text=f'{output}')







import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

