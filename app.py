from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np

# Load the trained model
model_path = 'model.pkl'  # Replace with your model's file path
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Initialize Flask app
app = Flask(__name__)

# Define the feature names (replace with your 10 selected features)
FEATURE_NAMES = [
    "Age", "ALB", "CA125", "HE4", "LYM#",
    "LYM%", "Menopouse", "NEU", "PCT", "PLT"
]

@app.route('/')
def index():
    return render_template('index.html', features=FEATURE_NAMES)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get feature values from the form
        feature_values = [float(request.form[f]) for f in FEATURE_NAMES]
        features = np.array(feature_values).reshape(1, -1)

        # Make prediction
        prediction = model.predict(features)
        probability = model.predict_proba(features)[:, 1]

        # Interpret prediction
        result = "High possibility of cancer" if prediction[0] == 1 else "Safe"
        return render_template(
            'index.html',
            features=FEATURE_NAMES,
            result=result,
            probability=f"{probability[0]:.2f}"
        )
    except Exception as e:
        print("Error:", e)  # Debugging output
        return render_template(
            'index.html',
            features=FEATURE_NAMES,
            error="An error occurred while processing your request."
        )

if __name__ == '__main__':
    app.run(debug=True)
