# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import mlflow
import mlflow.tracking
import pandas as pd
import numpy as np
import pickle
import os
from typing import List, Optional, Dict, Any # Added Dict, Any

# --- Configuration ---
# Set MLflow Tracking URI (from trainingflow.py)
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000/" # Replace if different
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Define Model URI (Using registered model from trainingflow.py/scoringflow.py)
MODEL_NAME = "churn_prediction_model_local" # Make sure this matches your registered model
MODEL_STAGE = "Staging" # Or "Production" if you promoted it
MODEL_URI = f"models:/{MODEL_NAME}/{MODEL_STAGE}" # <<< CORRECTED: Added stage

# Define path for downloaded scaler
SCALER_FILENAME = "scaler.pkl"
# --- End Configuration ---


# Create FastAPI app
app = FastAPI(title="Churn Prediction API (Updated Columns)")

# --- Data Models ---
# Updated based on the columns from the image
class PredictionRequest(BaseModel):
    # Numerical features
    CustomerID: int 
    Age: int
    Tenure: int
    UsageFrequency: float = Field(..., alias='Usage Frequency') # Use alias for space
    SupportCalls: int = Field(..., alias='Support Calls')
    PaymentDelay: int = Field(..., alias='Payment Delay')
    TotalSpend: float = Field(..., alias='Total Spend')
    LastInteraction: int = Field(..., alias='Last Interaction')

    # Categorical features
    Gender: str
    SubscriptionType: str = Field(..., alias='Subscription Type')
    ContractLength: str = Field(..., alias='Contract Length')

    class Config:
        allow_population_by_field_name = True # Allows using aliases like 'Usage Frequency'


# Define response data model for classification
class PredictionResponse(BaseModel):
    prediction: int # 0 or 1 for churn
    probability: Optional[float] = None # Probability of churn (class 1)
# --- End Data Models ---


# Global variables for model and scaler
model = None
scaler = None
preprocessing_features = None # To store feature names expected by scaler/pipeline step

# --- Startup Event ---
@app.on_event("startup")
async def load_model_and_scaler():
    global model, scaler, preprocessing_features
    print(f"Attempting to load model: {MODEL_URI}")
    try:
        # Load the MLflow model (Pipeline)
        model = mlflow.pyfunc.load_model(MODEL_URI)
        print("Model loaded successfully!")

        # --- Load the associated scaler ---
        # This assumes scaler is saved separately. If scaler is PART OF the MLflow
        # model pipeline, you might not need to load it separately.
        print("Attempting to load associated scaler (if saved separately)...")
        client = mlflow.tracking.MlflowClient()
        mv = client.get_latest_versions(MODEL_NAME, stages=[MODEL_STAGE])
        if not mv:
            print(f"Warning: No model version found for {MODEL_NAME}/{MODEL_STAGE} in MLflow registry. Cannot fetch associated scaler.")
            # If scaler is part of the pipeline, this might be okay.
        else:
            run_id = mv[0].run_id
            print(f"Found associated Run ID: {run_id}")

            # Define potential paths for the scaler artifact (adjust if needed)
            scaler_artifact_path_options = [
                 "scaler.pkl",
                 "preprocessing/scaler.pkl"
            ]
            scaler_local_path = None
            for artifact_path in scaler_artifact_path_options:
                try:
                    scaler_local_path = client.download_artifacts(run_id, artifact_path)
                    print(f"Scaler artifact found at run artifact path: {artifact_path}")
                    break # Stop searching once found
                except Exception as artifact_e:
                     print(f"Scaler not found at artifact path '{artifact_path}'. Trying next option...")

            if scaler_local_path and os.path.exists(scaler_local_path):
                with open(scaler_local_path, "rb") as f:
                    scaler = pickle.load(f)
                print(f"Scaler loaded successfully from {scaler_local_path}!")

                # --- Try to get feature names from the scaler ---
                if hasattr(scaler, 'feature_names_in_'):
                    preprocessing_features = list(scaler.feature_names_in_)
                    print(f"Scaler expects features: {preprocessing_features}")
                elif hasattr(scaler, 'get_feature_names_out'):
                     try:
                         preprocessing_features = list(scaler.get_feature_names_out())
                         print(f"Scaler generates features: {preprocessing_features}")
                     except Exception as e:
                         print(f"Could not get feature names via get_feature_names_out: {e}")
                else:
                     print("Warning: Could not automatically determine feature names from the scaler.")
            else:
                print("Info: Separate scaler artifact could not be downloaded or found. Assuming scaling is handled within the model pipeline if needed.")
                scaler = None # Ensure scaler is None if loading failed or not needed separately

    except Exception as e:
        print(f"Error during startup loading: {e}")
        model = None
        scaler = None
# --- End Startup Event ---


# --- Helper Functions ---
def preprocess_input(data: pd.DataFrame) -> pd.DataFrame:
    """
    Applies preprocessing steps.
    IMPORTANT: This function needs to mirror the logic in your training preprocessing.
               It might involve encoding, scaling, imputation etc.
               The current version makes assumptions - ADJUST AS NEEDED.
    """
    global scaler, preprocessing_features

    print(f"Preprocessing input data. Columns received: {list(data.columns)}")

    # --- Step 1: Handle Data Types / Missing Values (Example) ---
    # Ensure correct types, handle missing values according to your training logic
    # Example: If 'Total Spend' could be missing or non-numeric
    # if 'Total Spend' in data.columns:
    #    data['Total Spend'] = pd.to_numeric(data['Total Spend'], errors='coerce').fillna(0) # Example fillna

    # --- Step 2: Encoding Categorical Features (Placeholder/Assumption) ---
    # If your *saved MLflow model pipeline* already includes steps for encoding
    # (e.g., OneHotEncoder within a scikit-learn Pipeline), you might NOT need
    # to do explicit encoding here. The model.predict() call will handle it.
    #
    # If encoding needs to be done *before* a separate scaler is applied, you'd
    # load a fitted encoder artifact (like the scaler) and apply it here.
    categorical_features = ['Gender', 'Subscription Type', 'Contract Length'] # Based on new columns
    print(f"Info: Assuming encoding for {categorical_features} is handled by the MLflow model pipeline OR needs to be added here if not.")
    # Example (if needed):
    # if encoder is None: raise RuntimeError("Encoder not loaded")
    # encoded_data = encoder.transform(data[categorical_features])
    # encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(), index=data.index)
    # data = pd.concat([data.drop(columns=categorical_features), encoded_df], axis=1)

    # --- Step 3: Feature Selection / Ordering / Scaling ---
    # Option A: Scaling is done within the loaded MLflow pipeline
    if scaler is None:
        print("Info: No separate scaler loaded. Passing data directly to model pipeline (assuming it handles scaling if needed).")
        # Ensure columns match what the *pipeline's first step* expects.
        # You might need to know the expected column order/names from training.
        # If preprocessing_features were determined from an encoder instead:
        # if preprocessing_features: data = data[preprocessing_features]
        return data

    # Option B: Applying a separately loaded scaler
    else:
        print("Info: Applying separately loaded scaler.")
        if not preprocessing_features:
            # If feature names weren't inferred, assume scaler works on all numeric columns
            # Or define the list manually based on your training!
            numerical_cols = data.select_dtypes(include=np.number).columns.tolist()
            print(f"Warning: Scaler feature names not inferred. Assuming scaling applies to numeric columns: {numerical_cols}")
            preprocessing_features = numerical_cols # Use numeric columns

        try:
            # Select and reorder columns for the scaler
            data_to_scale = data[preprocessing_features]
            print(f"Data columns selected for scaler: {list(data_to_scale.columns)}")
        except KeyError as e:
            raise ValueError(f"Input data missing required feature for scaler: {e}. Required: {preprocessing_features}")

        try:
            # Apply the scaler
            scaled_values = scaler.transform(data_to_scale)
            scaled_df = pd.DataFrame(scaled_values, columns=data_to_scale.columns, index=data_to_scale.index)
            print("Data scaled successfully.")

            # Combine back with non-scaled columns if necessary (depends on your pipeline)
            # Example: If you only scaled numerical features and need to add back encoded categoricals
            # data = pd.concat([data.drop(columns=preprocessing_features), scaled_df], axis=1)
            # OR more likely, if scaler operated on ALL features after encoding:
            data = scaled_df # Replace original data with scaled data

            return data

        except Exception as e:
            raise RuntimeError(f"Error applying scaler: {e}")

# --- End Helper Functions ---


# --- API Endpoints ---
@app.get("/")
async def root():
    """Health check endpoint."""
    model_status = "loaded" if model else "not loaded"
    scaler_status = "loaded" if scaler else ("not loaded (or part of pipeline)" if model else "not loaded")
    return {
        "message": "Churn Prediction API is running",
        "model_status": model_status,
        "scaler_status": scaler_status
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make churn prediction based on updated columns."""
    global model # Scaler is used within preprocess_input if loaded

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    # Removed explicit check for scaler here, handled in preprocess_input

    try:
        # 1. Convert request Pydantic model to dictionary, using aliases
        input_dict = request.dict(by_alias=True)

        # 2. Create Pandas DataFrame (single row)
        input_df = pd.DataFrame([input_dict])
        # Optional: Reorder columns to a specific sequence if your model is sensitive to it
        # expected_order = ['Age', 'Gender', ...] # Define the exact order from training
        # input_df = input_df[expected_order]
        print(f"\nRaw input DataFrame:\n{input_df}")

        # 3. Preprocess the DataFrame
        # This step might handle encoding and scaling based on the helper function logic
        processed_df = preprocess_input(input_df.copy()) # Pass a copy
        print(f"\nProcessed DataFrame for prediction:\n{processed_df.head()}") # Print head for large #features

        # 4. Make prediction using the MLflow pipeline/model
        # The MLflow pyfunc model expects a DataFrame. It should match the input
        # format expected *after* any preprocessing steps done OUTSIDE the pipeline
        # (like the separate scaler if used), or the RAW format if the pipeline handles everything.
        prediction_result = model.predict(processed_df)

        # 5. Get probability if the underlying model supports it
        probability = None
        try:
            # Try to access predict_proba on the underlying model implementation
            if hasattr(model, '_model_impl') and hasattr(model._model_impl, 'predict_proba'):
                probabilities = model._model_impl.predict_proba(processed_df)
                probability = float(probabilities[0, 1]) # Prob of class 1
            elif hasattr(model, 'predict_proba'): # Direct access
                 probabilities = model.predict_proba(processed_df)
                 probability = float(probabilities[0, 1])
            else:
                print("Model type does not support probability predictions.")

            if probability is not None: print(f"Predicted probability (class 1): {probability:.4f}")

        except AttributeError:
             print("Could not access predict_proba method on the loaded model object.")
        except Exception as proba_e:
            print(f"Error getting probability: {proba_e}")


        # 6. Format response
        churn_prediction = int(prediction_result[0])
        print(f"Prediction result (class): {churn_prediction}")

        return PredictionResponse(prediction=churn_prediction, probability=probability)

    except ValueError as ve: # Catch specific errors like missing columns
         print(f"Data validation error: {str(ve)}")
         raise HTTPException(status_code=400, detail=f"Input data error: {str(ve)}")
    except RuntimeError as rte: # Catch errors from preprocessing steps
         print(f"Preprocessing/runtime error: {str(rte)}")
         raise HTTPException(status_code=500, detail=f"Preprocessing error: {str(rte)}")
    except Exception as e:
        # Log the full error for debugging on the server side
        print(f"Prediction endpoint error: {type(e).__name__} - {str(e)}")
        import traceback
        traceback.print_exc() # Print full traceback for server logs
        # Return a generic error to the client
        raise HTTPException(status_code=500, detail=f"Prediction failed due to an internal server error.")

# --- End API Endpoints ---


# Run the server if script is run directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)