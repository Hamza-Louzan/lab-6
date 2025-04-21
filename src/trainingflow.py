from metaflow import FlowSpec, step, Parameter
import mlflow
import os
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from hyperopt import STATUS_OK
import preprocessing  

mlflow.set_tracking_uri("http://127.0.0.1:5000")  # Example for remote server
mlflow.set_experiment("Customer_Churn_Pred_experiment")


class ChurnModelTrainingFlow(FlowSpec):
    """
    A flow for training customer churn prediction models with hyperparameter optimization.
    
    This flow:
    1. Preprocesses the data using the custom preprocessing module
    2. Evaluates different model types and hyperparameters
    3. Selects the best model
    4. Registers the model with MLflow
    """
    
    # Define parameters
    train_path = Parameter('train_path', 
                          help='Path to the training dataset',
                          default='../data/customer_churn_dataset-training-master.csv')
    
    test_path = Parameter('test_path',
                         help='Path to the testing dataset',
                         default='../data/test_set.csv')
    
    output_dir = Parameter('output_dir',
                          help='Directory to save processed data',
                          default='../data/processed')
    
    model_name = Parameter('model_name',
                          help='Name to register the model as in MLflow',
                          default='churn_prediction_model')
    
    random_state = Parameter('random_state',
                            help='Random seed for reproducibility',
                            default=42,
                            type=int)
    
    @step
    def start(self):
        """
        Start the flow and preprocess the data using the preprocessing module
        """
        print(f"Starting flow with training data from {self.train_path}")
        print(f"and testing data from {self.test_path}")
        
        # Use the preprocessing function from  module
        self.X_train, self.y_train, self.X_test, self.y_test, self.scaler= preprocessing.preprocess_churn_data(
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
                
        self.model_configs = [
            # Decision Tree configurations
            {'type': 'dt', 'max_depth': 5, 'min_samples_split': 2},
            {'type': 'dt', 'max_depth': 10, 'min_samples_split': 5},
            {'type': 'dt', 'max_depth': 15, 'criterion': 'entropy', 'min_samples_leaf': 3},

            
            # Random Forest configurations

            {'type': 'rf', 'n_estimators': 150, 'max_features': 'sqrt', 'class_weight': 'balanced'},
            {'type': 'rf', 'n_estimators': 300, 'max_depth': 12, 'min_samples_leaf': 2, 'bootstrap': True},
            
            # Logistic Regression configurations
            {'type': 'lr', 'C': 0.1, 'penalty': 'l2'},
            {'type': 'lr', 'max_iter': 1000},
            {'type': 'lr', 'C': 2.0, 'max_iter': 1000, 'solver': 'saga', 'penalty': 'elasticnet', 'l1_ratio': 0.5},
            
            # Gradient Boosting configurations
            {'type': 'gb', 'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3},
            {'type': 'gb', 'n_estimators': 150, 'learning_rate': 0.075, 'max_depth': 5, 'min_samples_leaf': 10},
            
            # XGBoost configurations
            {'type': 'xgb', 'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 4, 'subsample': 0.8},
            {'type': 'xgb', 'n_estimators': 200, 'learning_rate': 0.05, 'colsample_bytree': 0.8, 'gamma': 1},

            # KNN
            {'type': 'knn', 'n_neighbors': 3, 'weights': 'uniform'},
            {'type': 'knn', 'n_neighbors': 5, 'weights': 'distance', 'p': 1},
        ]
        
        print(f"Created {len(self.model_configs)} model configurations to evaluate")
        self.next(self.train_and_evaluate, foreach='model_configs')
    
    @step
    def train_and_evaluate(self):
        """
        Train a model with the current configuration and evaluate its performance
        """
        # Set up MLflow tracking
        data_dir = "/tmp/data"
        os.makedirs(data_dir, exist_ok=True)
        
        # Get the current model configuration
        params = self.input
        
        # Setup file paths for artifacts
        X_train_file = os.path.join(data_dir, "X_train.csv")
        y_train_file = os.path.join(data_dir, "y_train.csv")
        X_val_file = os.path.join(data_dir, "X_val.csv")
        y_val_file = os.path.join(data_dir, "y_val.csv")
        X_test_file = os.path.join(data_dir, "X_test.csv")
        y_test_file = os.path.join(data_dir, "y_test.csv")
        
        with mlflow.start_run(nested=True) as run:
            # Save datasets
            self.X_train.to_csv(X_train_file, index=False)
            self.y_train.to_csv(y_train_file, index=False)
            self.X_val.to_csv(X_val_file, index=False)
            self.y_val.to_csv(y_val_file, index=False)
            self.X_test.to_csv(X_test_file, index=False)
            self.y_test.to_csv(y_test_file, index=False)
            
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
                
            elif classifier_type == 'gb':
                from sklearn.ensemble import GradientBoostingClassifier
                clf = GradientBoostingClassifier(**model_params)
                model_name = "Gradient Boosting"
                
            elif classifier_type == 'xgb':
                import xgboost as xgb
                clf = xgb.XGBClassifier(**model_params)
                model_name = "XGBoost"
                
            elif classifier_type == 'svm':
                from sklearn.svm import SVC
                clf = SVC(**model_params)
                model_name = "Support Vector Machine"
                
            elif classifier_type == 'nn':
                from sklearn.neural_network import MLPClassifier
                clf = MLPClassifier(**model_params)
                model_name = "Neural Network"
                
            elif classifier_type == 'nb':
                from sklearn.naive_bayes import GaussianNB
                clf = GaussianNB(**model_params)
                model_name = "Naive Bayes"
                
            elif classifier_type == 'ada':
                from sklearn.ensemble import AdaBoostClassifier
                clf = AdaBoostClassifier(**model_params)
                model_name = "AdaBoost"
                
            elif classifier_type == 'knn':
                from sklearn.neighbors import KNeighborsClassifier
                clf = KNeighborsClassifier(**model_params)
                model_name = "K-Nearest Neighbors"
            
            else:
                raise ValueError(f"Unknown classifier type: {classifier_type}")
            
            # Create and fit pipeline
            pipeline = Pipeline([
                ('classifier', clf)
            ])
            
            pipeline.fit(self.X_train, self.y_train)
            
            # Calculate metrics for all sets
            train_acc = pipeline.score(self.X_train, self.y_train)
            val_acc = pipeline.score(self.X_val, self.y_val)
            test_acc = pipeline.score(self.X_test, self.y_test)
            
            # Log parameters and metrics
            mlflow.log_params(model_params)
            mlflow.set_tag("Model", model_name)
            mlflow.log_metric("train_accuracy", train_acc)
            mlflow.log_metric("validation_accuracy", val_acc)
            mlflow.log_metric("test_accuracy", test_acc)
            
            # Log model
            mlflow.sklearn.log_model(pipeline, "model")
            
            # Log datasets as artifacts
            mlflow.log_artifact(X_train_file, "datasets")
            mlflow.log_artifact(y_train_file, "datasets")
            mlflow.log_artifact(X_val_file, "datasets")
            mlflow.log_artifact(y_val_file, "datasets")
            mlflow.log_artifact(X_test_file, "datasets")
            mlflow.log_artifact(y_test_file, "datasets")
            
            # Log feature names
            feature_names = pd.DataFrame({'features': self.X_train.columns})
            feature_file = os.path.join(data_dir, "feature_names.csv")
            feature_names.to_csv(feature_file, index=False)
            mlflow.log_artifact(feature_file, "datasets")
            
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
    
    @step
    def join(self, inputs):
        """
        Compare all model results and select the best one
        """
        # Get all results
        self.all_results = [inp.result for inp in inputs]
        
        # Find the best model based on validation accuracy
        best_idx = min(range(len(self.all_results)), 
                      key=lambda i: self.all_results[i]['loss'])
        
        self.best_result = self.all_results[best_idx]
        
        # Preserve the scaler from the start step
        self.scaler = inputs[0].scaler
        
        print("\nBest Model Results:")
        print(f"Model Type: {self.best_result['model_type']}")
        print(f"Model Parameters: {self.best_result['model_params']}")
        print(f"Validation Accuracy: {self.best_result['val_acc']:.4f}")
        print(f"Test Accuracy: {self.best_result['test_acc']:.4f}")
        
        self.next(self.register_model)
    
    @step
    def register_model(self):
        """
        Register the best model with MLflow model registry
        """
        # Get the best model and its details
        best_model = self.best_result['model']
        model_type = self.best_result['model_type']
        params = self.best_result['model_params']
        metrics = {
            'train_accuracy': self.best_result['train_acc'],
            'validation_accuracy': self.best_result['val_acc'],
            'test_accuracy': self.best_result['test_acc']
        }
        
        # Register the model
        with mlflow.start_run() as run:
            # Log parameters
            for param_name, param_value in params.items():
                mlflow.log_param(param_name, param_value)
            mlflow.log_param('model_type', model_type)
            
            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Register the model
            mlflow.sklearn.log_model(
                best_model, 
                "model",
                registered_model_name=self.model_name
            )
            
            # Save the scaler as a pickle file to use during inference
            import pickle
            scaler_path = os.path.join("/tmp/data", "scaler.pkl")
            with open(scaler_path, "wb") as f:
                pickle.dump(self.scaler, f)
            
            # Log the scaler as an artifact
            mlflow.log_artifact(scaler_path, "preprocessing")
            
            self.final_run_id = run.info.run_id
        
        print(f"Model registered with MLflow as '{self.model_name}'")
        print(f"MLflow Run ID: {self.final_run_id}")
        
        self.next(self.end)
    
    @step
    def end(self):
        """
        End the flow and print final summary
        """
        print("\nChurn Model Training Flow Complete!")
        print(f"Best Model Type: {self.best_result['model_type']}")
        print(f"Best Model Parameters: {self.best_result['model_params']}")
        print(f"Train Accuracy: {self.best_result['train_acc']:.4f}")
        print(f"Validation Accuracy: {self.best_result['val_acc']:.4f}")
        print(f"Test Accuracy: {self.best_result['test_acc']:.4f}")
        print(f"Model registered as: {self.model_name}")
        print(f"Final MLflow Run ID: {self.final_run_id}")

if __name__ == '__main__':
    ChurnModelTrainingFlow()