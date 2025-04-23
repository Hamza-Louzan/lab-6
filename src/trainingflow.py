from metaflow import FlowSpec, step, Parameter, kubernetes, timeout, retry, catch, conda, resources
import mlflow
import os
import pandas as pd
import numpy as np
from hyperopt import STATUS_OK
import preprocessing  

# Set the MLflow tracking URI
# Set the MLflow tracking URI first
mlflow.set_tracking_uri('http://35.235.91.231:5000')
# Then set the experiment
mlflow.set_experiment("Customer_Churn_Pred_GCP_experiment")

class ChurnModelTrainingFlowGCP(FlowSpec):
    """
    A flow for training customer churn prediction models with hyperparameter optimization,
    designed to run on Kubernetes.
    
    This flow:
    1. Preprocesses the data using the custom preprocessing module
    2. Evaluates different model types and hyperparameters
    3. Selects the best model
    4. Registers the model with MLflow
    """
    
    # Define parameters with local file paths
    train_path = Parameter('train_path', 
                         help='Path to the training dataset',
                         default='gs://lab7-hl/data/customer_churn_dataset-training-master.csv')  # Update with your local path
    
    test_path = Parameter('test_path',
                        help='Path to the testing dataset',
                        default='gs://lab7-hl/data/test_set.csv')  # Update with your local path
    
    output_dir = Parameter('output_dir',
                         help='Directory to save processed data',
                         default='gs://lab7-hl/data')  # Update with your local output directory
    
    model_name = Parameter('model_name',
                         help='Name to register the model as in MLflow',
                         default='churn_prediction_model')  # Optional: Customize your model name
    
    random_state = Parameter('random_state',
                           help='Random seed for reproducibility',
                           default=42,
                           type=int)
    
    @kubernetes(service_account="lab7-mlflow-acct", image="gcr.io/lab7-457620/image2:1.1")
    @resources(cpu=1, memory=4096)
    @timeout(minutes=15)
    @retry(times=2)
    @step
    def start(self):
        """
        Start the flow and preprocess the data using the preprocessing module
        """
        print(f"Starting flow with training data from {self.train_path}")
        print(f"and testing data from {self.test_path}")
        
        # Check for errors
        if hasattr(self, 'start_error') and self.start_error:
            print(f"Error in start step: {self.start_error}")
            raise ValueError("Failed to start flow")
            
        # Use the preprocessing function from module
        self.X_train, self.y_train, self.X_test, self.y_test, self.scaler = preprocessing.preprocess_churn_data(
            train_path=self.train_path,
            test_path=self.test_path,
            output_dir=self.output_dir
        )
        
        if self.X_train is None:
            raise ValueError("Preprocessing failed. Check file paths and data format.")
        
        print(f"Preprocessing complete. Training data shape: {self.X_train.shape}")
        
        # Create validation set from training data
        from sklearn.model_selection import train_test_split
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X_train, self.y_train, test_size=0.2, random_state=self.random_state
        )
                
        # Define model configurations to evaluate
        self.model_configs = [
            # Decision Tree configurations
            {'type': 'dt', 'max_depth': 5, 'min_samples_split': 2},
            {'type': 'dt', 'max_depth': 10, 'min_samples_split': 5},
            
            # Random Forest configurations
            {'type': 'rf', 'n_estimators': 150, 'max_features': 'sqrt', 'class_weight': 'balanced'},
            {'type': 'rf', 'n_estimators': 300, 'max_depth': 12, 'min_samples_leaf': 2, 'bootstrap': True},
            
            # Logistic Regression configurations
            {'type': 'lr', 'C': 0.1, 'penalty': 'l2'},
            {'type': 'lr', 'C': 2.0, 'max_iter': 1000, 'solver': 'saga', 'penalty': 'elasticnet', 'l1_ratio': 0.5},
            
            # XGBoost configurations
            {'type': 'xgb', 'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 4, 'subsample': 0.8},
        ]
        
        print(f"Created {len(self.model_configs)} model configurations to evaluate")
        self.next(self.train_and_evaluate, foreach='model_configs')
    
    @kubernetes(service_account="lab7-mlflow-acct", image="gcr.io/lab7-457620/image2:1.1")
    @timeout(minutes=30)
    @retry(times=2)
    @step
    def train_and_evaluate(self):
        """
        Train a model with the current configuration and evaluate its performance
        """
        # Check for errors
        if hasattr(self, 'train_error') and self.train_error:
            print(f"Error in training step: {self.train_error}")
            self.result = {
                'loss': float('inf'),
                'status': 'fail',
                'model_type': 'error'
            }
            self.next(self.join)
            return
            
        # Get the current model configuration
        params = self.input
        
        with mlflow.start_run(nested=True) as run:
            print(f"MLflow run started with ID: {run.info.run_id}")
            
            # Model Selection
            classifier_type = params['type']
            model_params = params.copy()
            del model_params['type']
            
            # Initialize clf and model_name variables
            clf = None
            model_name = "Unknown"
            
            # Import necessary model packages based on classifier type
            if classifier_type == 'dt':
                from sklearn.tree import DecisionTreeClassifier
                clf = DecisionTreeClassifier(**model_params)
                model_name = "Decision Tree"
                
            elif classifier_type == 'rf':
                from sklearn.ensemble import RandomForestClassifier
                clf = RandomForestClassifier(**model_params)
                model_name = "Random Forest"
                
            elif classifier_type == 'lr':
                from sklearn.linear_model import LogisticRegression
                penalty = model_params.get('penalty')
                if penalty == 'l1':
                    model_params['solver'] = 'saga'
                elif penalty == 'l2':
                    model_params['solver'] = 'lbfgs'
                elif penalty is None:
                    model_params['solver'] = 'lbfgs'
                clf = LogisticRegression(**model_params)
                model_name = "Logistic Regression"
                
            elif classifier_type == 'xgb':
                import xgboost as xgb
                clf = xgb.XGBClassifier(**model_params)
                model_name = "XGBoost"
            
            else:
                raise ValueError(f"Unknown classifier type: {classifier_type}")
            
            # Create and fit pipeline
            from sklearn.pipeline import Pipeline
            pipeline = Pipeline([
                ('classifier', clf)
            ])
            
            print(f"Training {model_name} model...")
            pipeline.fit(self.X_train, self.y_train)
            
            # Calculate metrics for all sets
            train_acc = pipeline.score(self.X_train, self.y_train)
            val_acc = pipeline.score(self.X_val, self.y_val)
            test_acc = pipeline.score(self.X_test, self.y_test)
            
            # Log parameters and metrics
            mlflow.log_params(model_params)
            mlflow.set_tag("Model", model_name)
            mlflow.set_tag("Executed_on", "Local Machine")
            mlflow.log_metric("train_accuracy", train_acc)
            mlflow.log_metric("validation_accuracy", val_acc)
            mlflow.log_metric("test_accuracy", test_acc)
            
            # Log model
            mlflow.sklearn.log_model(pipeline, "model")
            
            print(f"\nModel: {model_name}")
            print(f"Train Accuracy: {train_acc:.4f}")
            print(f"Validation Accuracy: {val_acc:.4f}")
            print(f"Test Accuracy: {test_acc:.4f}")
            
            # Store the results and model
            self.result = {
                'loss': -val_acc,  # Negative because we want to maximize accuracy
                'status': STATUS_OK,
                'model': pipeline,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'test_acc': test_acc,
                'model_type': classifier_type,
                'model_params': model_params,
                'run_id': run.info.run_id
            }
            
        self.next(self.join)
    
    @kubernetes(service_account="lab7-mlflow-acct", image="gcr.io/lab7-457620/image2:1.1")
    @timeout(minutes=10)
    @retry(times=2)
    @step
    def join(self, inputs):
        """
        Compare all model results and select the best one
        """
        # Check for errors
        if hasattr(self, 'join_error') and self.join_error:
            print(f"Error in join step: {self.join_error}")
            raise ValueError("Failed to join results")
            
        # Get all results
        self.all_results = [inp.result for inp in inputs]
        
        # Filter out any failed runs
        valid_results = [r for r in self.all_results if r.get('status') != 'fail']
        
        if not valid_results:
            raise ValueError("All model training runs failed")
        
        # Find the best model based on validation accuracy
        best_idx = min(range(len(valid_results)), 
                      key=lambda i: valid_results[i]['loss'])
        
        self.best_result = valid_results[best_idx]
        
        # Preserve the scaler from the start step
        self.scaler = inputs[0].scaler
        
        print("\nBest Model Results:")
        print(f"Model Type: {self.best_result['model_type']}")
        print(f"Model Parameters: {self.best_result['model_params']}")
        print(f"Validation Accuracy: {self.best_result['val_acc']:.4f}")
        print(f"Test Accuracy: {self.best_result['test_acc']:.4f}")
        
        self.next(self.register_model)
    
    @kubernetes(service_account="lab7-mlflow-acct", image="gcr.io/lab7-457620/image2:1.1")
    @timeout(minutes=15)
    @retry(times=2)
    @step
    def register_model(self):
        """
        Register the best model with MLflow model registry
        """
        # Check for errors
        if hasattr(self, 'register_error') and self.register_error:
            print(f"Error registering model: {self.register_error}")
            raise ValueError("Failed to register model")
            
        # Get the best model and its details
        best_model = self.best_result['model']
        model_type = self.best_result['model_type']
        params = self.best_result['model_params']
        metrics = {
            'train_accuracy': self.best_result['train_acc'],
            'validation_accuracy': self.best_result['val_acc'],
            'test_accuracy': self.best_result['test_acc']
        }
        
        # Register the model with MLflow
        with mlflow.start_run() as run:
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(best_model, "model")
            print(f"Model registered successfully in MLflow under run ID: {run.info.run_id}")
            print(f"Model Type: {model_type}")
            print(f"Model Parameters: {params}")
        
        self.next(self.end)
    
    @step
    def end(self):
        """
        End the flow
        """
        print("Training and model registration complete.")

# Run the flow
if __name__ == '__main__':
    ChurnModelTrainingFlowGCP()
