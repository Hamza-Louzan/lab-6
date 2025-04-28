# test_app.py
import requests
import json
import sys

# API endpoint
url = "http://localhost:8000/predict"
health_url = "http://localhost:8000/"

# --- Test Data ---
# Updated based on the new columns from the image.
# IMPORTANT: Replace these values with realistic examples relevant to your model.
#            Keys MUST match the field names OR the aliases defined in app.py.
test_data = {
    # Numerical features
    "CustomerID": 12345,          
    "Age": 35,
    "Tenure": 24,
    "Usage Frequency": 15.5,        # Uses alias
    "Support Calls": 2,             # Uses alias
    "Payment Delay": 5,             # Uses alias
    "Total Spend": 1500.75,         # Uses alias
    "Last Interaction": 30,         # Uses alias

    # Categorical features
    "Gender": "0",
    "Subscription Type": "0", # Uses alias
    "Contract Length": "1"     # Uses alias
}
# --- End Test Data ---


def test_prediction():
    """Tests the /predict endpoint of the Churn Prediction API (Updated)."""
    try:
        # 1. Health Check
        print(f"Checking API health at: {health_url}")
        health_response = requests.get(health_url, timeout=5)
        print(f"Health check status code: {health_response.status_code}")
        health_response.raise_for_status()
        print(f"Health check response: {health_response.json()}")

        health_info = health_response.json()
        if health_info.get("model_status") != "loaded":
             print("Warning: Model reported as 'not loaded' by the health check.")
             # Decide if you want to stop the test here
             # return

        # 2. Make Prediction Request
        print(f"\nSending request to: {url}")
        print(f"Request data:\n{json.dumps(test_data, indent=2)}")
        response = requests.post(url, json=test_data, timeout=10)

        # 3. Print Response
        print(f"\nResponse Status Code: {response.status_code}")

        if response.ok:
            print(f"Response JSON:\n{json.dumps(response.json(), indent=2)}")
        else:
            try:
                 error_detail = response.json().get('detail', response.text)
            except json.JSONDecodeError:
                 error_detail = response.text
            print(f"Error: {error_detail}")


    except requests.exceptions.Timeout:
         print(f"Error: Request timed out connecting to {health_url} or {url}")
    except requests.exceptions.ConnectionError:
        print(f"Error: Failed to connect to the API. Make sure it's running at http://localhost:8000")
    except requests.exceptions.RequestException as e:
         print(f"Error during request: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    test_prediction()