from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load trained models and scaler
with open('naive_bayes_model.pkl', 'rb') as file:
    naive_bayes_model = pickle.load(file)

with open('perceptron_model.pkl', 'rb') as file:
    perceptron_model = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        model_type = data.get("model_type", "naive_bayes")

        # Extract features from request data
        glucose = data.get("glucose")
        insulin = data.get("insulin")
        bmi = data.get("bmi")
        age = data.get("age")

        # Ensure all required fields are present
        if None in [glucose, insulin, bmi, age]:
            return jsonify({"error": "Missing one or more required fields"}), 400

        # Prepare the input data for scaling and prediction
        input_data = [[glucose, insulin, bmi, age]]

        # Apply standard scaling
        scaled_input_data = scaler.transform(input_data)

        # Select the model based on model_type parameter
        if model_type == "perceptron":
            model = perceptron_model
        else:
            model = naive_bayes_model

        # Make prediction
        prediction = model.predict(scaled_input_data)[0]

        # Convert prediction to a Python int for JSON serialization
        prediction = int(prediction)

        # Return the result as JSON
        return jsonify({"predicted_diabetes_type": prediction})

    except Exception as e:
        print(f"Error: {e}")  # Print error to console for debugging
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
