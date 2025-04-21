from metaflow import FlowSpec, step, Parameter, Flow
import mlflow
import os
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
import pickle

class ChurnScoringFlow(FlowSpec):
    """
    A flow for scoring using a trained churn prediction model.
    
    This flow:
    1. Loads a registered model from MLflow
    2. Loads and preprocesses new data
    3. Makes predictions on the new data
    4. Saves the prediction results
    """
    
    # Define parameters
    model_name = Parameter('model_name',
                          help='Name of the registered model in MLflow',
                          default='churn_prediction_model')
    
    model_stage = Parameter('model_stage',
                           help='Stage of the model to use (Production, Staging, etc.)',
                           default='Production')
    
    data_path = Parameter('data_path',
                         help='Path to the data for prediction',
                         default='../data/holdout_set.csv')
    
    output_path = Parameter('output_path',
                           help='Path to save prediction results',
                           default='../data/churn_predictions.csv')
    
    @step
    def start(self):
        """
        Start the flow and load the model and scaler
        """
        print(f"Starting churn prediction flow with model: {self.model_name} ({self.model_stage})")
        
        # Try to load the registered model from MLflow
        try:
            model_uri = f"models:/{self.model_name}/{self.model_stage}"
            self.model = mlflow.sklearn.load_model(model_uri)
            print(f"Loaded model: {self.model}")
            
            # Get the run ID that created this model version
            import mlflow.tracking
            client = mlflow.tracking.MlflowClient()
            model_details = client.get_registered_model(self.model_name)
            latest_version = model_details.latest_versions[0]
            run_id = latest_version.run_id
            
            # Get the scaler from the same run's artifacts
            artifact_path = client.download_artifacts(run_id, "preprocessing/scaler.pkl")
            with open(artifact_path, "rb") as f:
                self.scaler = pickle.load(f)
            print("Loaded scaler from MLflow artifacts")
            
        except Exception as e:
            print(f"Error loading model from MLflow registry: {e}")
            print("Trying to load from latest training flow instead...")
            
            # Alternatively, get model and scaler from the latest training flow
            try:
                latest_train_run = Flow('ChurnModelTrainingFlow').latest_run
                self.model = latest_train_run['register_model'].task.data.best_result['model']
                self.scaler = latest_train_run['register_model'].task.data.scaler
                print("Retrieved model and scaler from latest training flow")
            except Exception as e:
                print(f"Error retrieving model from training flow: {e}")
                raise
        
        self.next(self.load_data)
    
    @step
    def load_data(self):
        """
        Load new data for prediction
        """
        print(f"Loading data from {self.data_path}")
        
        # Load the new data
        try:
            self.raw_data = pd.read_csv(self.data_path)
            print(f"Loaded data with shape: {self.raw_data.shape}")
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
        
        # Check if the target column exists
        if 'Churn' in self.raw_data.columns:
            self.has_target = True
            self.target_values = self.raw_data['Churn']
            self.data = self.raw_data.drop(columns=['Churn'])
            print("Found target column 'Churn'. Will evaluate prediction performance.")
        else:
            self.has_target = False
            self.data = self.raw_data
            print("No target column found. Will only make predictions.")
        
        self.next(self.preprocess_data)
                
    @step
    def preprocess_data(self):
        """
        Preprocess the new data using the same preprocessing module as training
        """
        print("Preprocessing new data...")
        
        import preprocessing
        import tempfile
        import os
        
        # Create temporary files
        temp_dir = tempfile.mkdtemp()
        temp_train_path = os.path.join(temp_dir, 'temp_train.csv')
        temp_test_path = os.path.join(temp_dir, 'temp_test.csv')
        
        # Create temp_data outside the try block so it's available in all code paths
        if self.has_target:
            temp_data = self.raw_data.copy()
        else:
            temp_data = self.data.copy()
            temp_data['Churn'] = 0  # Dummy value
        
        # First, we need to load the original training data to get the same feature selection
        try:
            latest_train_run = Flow('ChurnModelTrainingFlow').latest_run
            original_train_path = latest_train_run['start'].task.train_path
            print(f"Using original training data path: {original_train_path}")
            
            # Save to the test file
            temp_data.to_csv(temp_test_path, index=False)
            
            # Call the preprocessing function with original training data
            # This returns X_train, y_train, X_processed, y_processed, scaler
            _, _, X_processed, y_processed, _ = preprocessing.preprocess_churn_data(
                train_path=original_train_path,  # Use original training data
                test_path=temp_test_path,        # Use new data as test
                output_dir=temp_dir
            )
            
            # Store both processed features and targets
            self.data = X_processed
            
            # Update target values if we have them
            if self.has_target:
                self.target_values = y_processed
            
            print(f"Preprocessing complete. Processed data shape: {self.data.shape}")
            print(f"Selected features: {list(self.data.columns)}")
            
        except Exception as e:
            print(f"Error using original training data: {e}")
            print("Falling back to default preprocessing approach")
            
            # Save both train and test from current data
            temp_data.to_csv(temp_train_path, index=False)
            temp_data.to_csv(temp_test_path, index=False)
            
            # Fall back to processing with just current data
            # This returns X_train, y_train, X_processed, y_processed, scaler
            X_processed, _, _, y_processed, _ = preprocessing.preprocess_churn_data(
                train_path=temp_train_path,
                test_path=temp_test_path,
                output_dir=temp_dir
            )
            
            self.data = X_processed
            
            # Update target values if we have them
            if self.has_target:
                self.target_values = y_processed
        
        self.next(self.make_predictions)

    
    @step
    def make_predictions(self):
        """
        Make predictions using the model
        """
        print("Making churn predictions...")
        
        # Make predictions
        self.predictions = self.model.predict(self.data)
        
        # Get probability predictions if the model supports it
        if hasattr(self.model, 'predict_proba'):
            try:
                self.probabilities = self.model.predict_proba(self.data)
                print("Generated probability predictions")
            except:
                self.probabilities = None
                print("Could not generate probability predictions")
        else:
            self.probabilities = None
        
        # If we have true churn values, evaluate the predictions
        if self.has_target:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
            
            self.metrics = {
                'accuracy': accuracy_score(self.target_values, self.predictions),
                'precision': precision_score(self.target_values, self.predictions, average='weighted'),
                'recall': recall_score(self.target_values, self.predictions, average='weighted'),
                'f1': f1_score(self.target_values, self.predictions, average='weighted')
            }
            
            # Create confusion matrix
            self.confusion_matrix = confusion_matrix(self.target_values, self.predictions)
            
            print("\nPrediction Performance:")
            for metric_name, metric_value in self.metrics.items():
                print(f"  {metric_name}: {metric_value:.4f}")
            
            print("\nConfusion Matrix:")
            print(self.confusion_matrix)
        
        self.next(self.save_results)
    
    @step
    def save_results(self):
        """
        Save prediction results to file
        """
        print(f"Saving churn predictions to {self.output_path}")
        
        # Create a DataFrame with predictions
        # Start with original data
        results = self.raw_data.copy()
        
        # Add predictions
        results['Predicted_Churn'] = self.predictions
        
        # Add probabilities if available
        if self.probabilities is not None:
            if self.probabilities.shape[1] >= 2:  # Binary classification
                results['Churn_Probability'] = self.probabilities[:, 1]  # Probability of class 1 (churn)
        
        # Ensure output directory exists
        output_dir = os.path.dirname(self.output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Save to CSV
        results.to_csv(self.output_path, index=False)
        self.results = results
        
        print(f"Saved predictions to {self.output_path}")
        self.next(self.end)
    
    @step
    def end(self):
        """
        End the flow and print summary
        """
        print("\nChurn Prediction Flow Complete!")
        print(f"Predictions saved to: {self.output_path}")
        
        # Print sample of predictions
        print("\nSample predictions (first 5 rows):")
        print(self.results.head())
        
        # If we evaluated performance, print the metrics again
        if self.has_target:
            print("\nPrediction Performance Metrics:")
            for metric_name, metric_value in self.metrics.items():
                print(f"  {metric_name}: {metric_value:.4f}")

if __name__ == '__main__':
    ChurnScoringFlow()