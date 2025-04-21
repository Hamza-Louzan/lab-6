import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

def preprocess_churn_data(train_path='../data/customer_churn_dataset-training-master.csv',
                          test_path='../data/customer_churn_dataset-testing-master.csv',
                          output_dir='../data/processed'):
    """
    Preprocess customer churn datasets for machine learning.
    Parameters:
    -----------
    train_path : str
        Path to the training dataset CSV file
    test_path : str
        Path to the testing dataset CSV file
    output_dir : str
        Directory to save processed data files
    Returns:
    --------
    tuple
        (X_train, y_train, X_test, y_test, scaler)
    """
    # Load the datasets
    try:
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print(f"Please ensure the files exist at the specified paths")
        return None, None, None, None, None

    # Handle missing values in both train and test sets
    train = train.dropna(axis=0)
    # test = test.dropna(axis=0)  
    
    # Split into features and target
    X_train = train.drop(columns='Churn')
    y_train = train['Churn']
    X_test = test.drop(columns='Churn')
    y_test = test['Churn']
    
    # Map categorical variables
    gender_map = {'Male': 0, 'Female': 1}
    subscription_map = {'Basic': 0, 'Premium': 2, 'Pro': 3, 'Standard': 1}
    contract_map = {'Annual': 0, 'Quarterly': 1, 'Monthly': 2}
    
    # Apply mappings to train data
    X_train['Gender'] = X_train['Gender'].map(gender_map)
    X_train['Subscription Type'] = X_train['Subscription Type'].map(subscription_map)
    X_train['Contract Length'] = X_train['Contract Length'].map(contract_map)
    
    # Apply mappings to test data
    X_test['Gender'] = X_test['Gender'].map(gender_map)
    X_test['Subscription Type'] = X_test['Subscription Type'].map(subscription_map)
    X_test['Contract Length'] = X_test['Contract Length'].map(contract_map)
    
    # Check for any missing values after mapping (in case of unknown categories)
    X_train = X_train.dropna(axis=0)  # Added this line to handle NaNs after mapping
    X_test = X_test.dropna(axis=0)  # Added this line to handle NaNs after mapping
    
    # Make sure y_train and y_test match the filtered X dataframes
    y_train = y_train.loc[X_train.index]
    y_test = y_test.loc[X_test.index]
    
    # Scale numeric features
    numeric_features = X_train.select_dtypes(include=['float64', 'int64']).columns
    scaler = StandardScaler()
    X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
    X_test[numeric_features] = scaler.transform(X_test[numeric_features])
    
    # Reset indices after all the filtering
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save processed data
    X_train.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
    y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
    X_test.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
    y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)
    
    return X_train, y_train, X_test, y_test, scaler

if __name__ == "__main__":
    preprocess_churn_data()