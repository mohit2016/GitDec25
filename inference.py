import joblib
import numpy as np

# Load the trained Decision Tree Classification model
model = joblib.load('dt_classification_model.pkl')

# Example: Prepare input data for prediction
# Replace with your actual feature data
X_new = np.array([[5.1, 3.5, 1.4, 0.2]])  # Example feature values

# Make predictions
predictions = model.predict(X_new)
print(f"Prediction: {predictions[0]}")

# Get prediction probabilities (if needed)
probabilities = model.predict_proba(X_new)
print(f"Prediction probabilities: {probabilities[0]}")