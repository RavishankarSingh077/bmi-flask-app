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
    # Get data from form
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]

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
        output = 'Obese Class 1 (First level of obesity.)'

    return render_template('index.html', prediction_text=f'{output}')


if __name__ == "__main__":
    app.run(debug=True)
