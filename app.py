from flask import Flask, render_template, request
import pickle
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from sklearn.preprocessing import StandardScaler
import os

# Set base path
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'training'))

# Define file paths
model_weights_path = os.path.join(base_path, 'model.weights.h5')
model_architecture_path = os.path.join(base_path, 'model_architecture.json')
scaler_path = os.path.join(base_path, 'scaler.pkl')
choices_path = os.path.join(base_path, 'choices.pkl')
columns_list_path = os.path.join(base_path, 'columns_list.csv')

# Create the Flask app
app = Flask(__name__)

# Load the list of expected columns from the CSV
def load_expected_columns(file_path):
    # Read the CSV file into a list
    df = pd.read_csv(file_path, header=None)
    return [col for col in df[0]]

# Load expected columns
expected_columns = load_expected_columns(columns_list_path)
expected_cat_cols = [col for col in expected_columns if '_cat' in col]
expected_cont_cols = [col for col in expected_columns if '_cont' in col]

# Load the model architecture and weights
with open(model_architecture_path, 'r') as f:
    model_architecture = f.read()

model = model_from_json(model_architecture)
model.load_weights(model_weights_path)

# Load the scaler and choices
with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

with open(choices_path, 'rb') as f:
    choices = pickle.load(f)

@app.route("/", methods=['GET', 'POST'])
def index():
    ''' 
        Function to return result only if POST request
    '''
    if request.method == 'POST':
        input_data = request.form.to_dict()
        result = predict(input_data)
        return render_template('index.html', choices=choices, result=result)
    else: 
        return render_template('index.html', choices=choices)

def predict(input_data):
    '''
        Helper function for making prediction
    '''
    # Extract values from input_data
    combined_income = float(input_data.get("Combined_income", 0))
    requested_amount = float(input_data.get("Requested_amount", 0))
    credit_history = 1 if input_data.get("Credit_history") == "Yes" else 0
    community_type = input_data.get("Community_type")
    dependents = input_data.get("Dependents")
    college_degree = input_data.get("College_degree")

    # Create a DataFrame for input data
    input_df = pd.DataFrame([{
        "Combined_income": combined_income,
        "Requested_amount": requested_amount,
        "Credit_history": credit_history,
        "Community_type": community_type,
        "Dependents": dependents,
        "College_degree": college_degree
    }])

    # Perform one-hot encoding
    input_df_encoded = pd.get_dummies(input_df, columns=['Community_type', 'Dependents', 'College_degree'])

    # Ensure all expected columns are present
    for col in expected_columns:
        if col not in input_df_encoded.columns:
            input_df_encoded[col] = 0

    # Reindex to ensure column order matches
    input_df_encoded = input_df_encoded.reindex(columns=expected_columns, fill_value=0)

    # Prepare continuous features
    X_cont = input_df_encoded[expected_cont_cols].astype('float')

    # Scale continuous features
    X_cont_scaled = scaler.transform(X_cont)

    # Combine scaled continuous features and categorical features
    X = pd.concat([pd.DataFrame(X_cont_scaled, columns=expected_cont_cols), input_df_encoded[expected_cat_cols]], axis=1)

    # Make prediction
    output = model.predict(X)

    return output[0]

if __name__ == '__main__':
    app.run(debug=True)


