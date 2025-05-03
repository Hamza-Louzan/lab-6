from fastapi import FastAPI
import uvicorn
import joblib
from pydantic import BaseModel
import numpy as np # Make sure numpy is imported if predict_proba returns numpy array

app = FastAPI(
    title="Reddit Comment Classifier",
    description="Classify Reddit comments as either 1 = Remove or 0 = Do Not Remove.",
    version="0.1",
)

# Defining the request body structure using Pydantic
class request_body(BaseModel):
    reddit_comment : str

# Load the model pipeline during startup
@app.on_event('startup')
def load_artifacts():
    global model_pipeline
    # Ensure the joblib file is in the same directory or provide the correct path
    model_pipeline = joblib.load("reddit_model_pipeline.joblib")
    print("Model pipeline loaded successfully.") # Optional: confirmation message

# Defining path operation for root endpoint
@app.get('/')
def main():
    return {'message': 'This is a model for classifying Reddit comments'}

# Defining path operation for /predict endpoint
@app.post('/predict')
def predict(data : request_body):
    try:
        X = [data.reddit_comment] # Model expects an iterable (like a list)
        # Use predict_proba as shown in Lab 8 to get probabilities
        predictions_proba = model_pipeline.predict_proba(X)
        # Extract probabilities for class 1 (Remove) and class 0 (Keep)
        # Assuming class 1 is the second column
        # Convert numpy array to list for JSON serialization
        predictions_list = predictions_proba.tolist()
        return {'Predictions': predictions_list}
    except Exception as e:
        # Log the error for debugging
        print(f"Error during prediction: {e}")
        # Return an error response
        return {"error": "Prediction failed", "details": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)