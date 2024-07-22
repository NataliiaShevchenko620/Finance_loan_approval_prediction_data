import pandas as pd
import numpy as np
import pickle
from flask import Flask, request, render_template

# Load the model, scaler, and choices dictionary
with open('training/model.pkl', 'rb') as f: 
    model = pickle.load(f)
with open('training/scaler.pkl', 'rb') as f: 
    scaler = pickle.load(f)
with open('training/choices.pkl', 'rb') as f: 
    choices = pickle.load(f)

app = Flask(__name__)

# Define the categorical and continuous columns
cat_cols = [col for col in choices.keys() if choices[col] is not None]
cont_cols = [col for col in choices.keys() if choices[col] is None]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        input_data = {
            'Combined_income': request.form.get('Combined_income', type=int),
            'Credit_history': request.form.get('Credit_history'),
            'Requested_amount': request.form.get('Requested_amount', type=int),
            'Community_type': request.form.get('Community_type'),
            'Dependents': request.form.get('Dependents'),
            'College_degree': request.form.get('College_degree')
        }
        result = predict(input_data)
        return render_template('index.html', result=result)
    return render_template('index.html')


@app.route("/tableau")
def tableau():
    print('here')
    """
    Route to render the Tableau dashboard
    """
    return render_template('tableau.html')


def predict(input_data):
    """
    Helper function to make predictions based on input data.
    """
    # Create a DataFrame with the correct columns
    X = pd.DataFrame(columns=choices.keys())
    
    # Map input data to the correct columns
    X.loc[0, 'ApplicantIncome'] = input_data['Combined_income'] * 0.6  # Example split
    X.loc[0, 'CoapplicantIncome'] = input_data['Combined_income'] * 0.4  # Example split
    X.loc[0, 'LoanAmount'] = input_data['Requested_amount']
    X.loc[0, 'Loan_Amount_Term'] = 360  # Example value, adjust as necessary
    
    # Handle categorical variables
    X.loc[0, 'Credit_History'] = 1 if input_data['Credit_history'] == 'Yes' else 0
    X.loc[0, 'Dependents_0'] = 1 if input_data['Dependents'] == '0' else 0
    X.loc[0, 'Dependents_1'] = 1 if input_data['Dependents'] == '1' else 0
    X.loc[0, 'Dependents_2'] = 1 if input_data['Dependents'] == '2' else 0
    X.loc[0, 'Dependents_3+'] = 1 if input_data['Dependents'] == '3+' else 0
    X.loc[0, 'Education_Graduate'] = 0 if input_data['College_degree'] == 'No' else 1
    X.loc[0, 'Education_Not Graduate'] = 1 if input_data['College_degree'] == 'No' else 0
    X.loc[0, 'Property_Area_Urban'] = 1 if input_data['Community_type'] == 'Urban' else 0
    X.loc[0, 'Property_Area_Semiurban'] = 1 if input_data['Community_type'] == 'Suburban' else 0
    X.loc[0, 'Property_Area_Rural'] = 1 if input_data['Community_type'] == 'Rural' else 0
    
    # Ensure all columns are present
    X = X.fillna(0)
    
    # Scale input data
    X_transformed = scaler.transform(X)
    
    # Make prediction
    output = model.predict(X_transformed)
    
    # Apply threshold to determine binary class
    threshold = 0.5
    prediction = (output[0] > threshold).astype(int)
    
    # Return 'No' for 0 and 'Yes' for 1
    return 'Yes' if prediction == 1 else 'No'

if __name__ == '__main__':
    app.run(debug=True)
