from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load model, scaler, and choices dictionary
with open('training/model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('training/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('training/choices.pkl', 'rb') as f:
    choices = pickle.load(f)

# Define categorical and continuous columns
cat_cols = [col for col in choices.keys() if choices[col] is not None]
cont_cols = [col for col in choices.keys() if choices[col] is None]

def predict(input_data): 
    """
    Helper function to make predictions based on input data.
    """
    # Create a DataFrame with the correct columns
    X = pd.DataFrame(columns=choices.keys())
    
    # Map input data to the correct columns
    X.loc[0, 'ApplicantIncome'] = input_data.get('Combined_income', np.nan)
    X.loc[0, 'CoapplicantIncome'] = 0  # Set CoapplicantIncome to 0 explicitly
    X.loc[0, 'LoanAmount'] = input_data.get('Requested_amount', np.nan)
    X.loc[0, 'Loan_Amount_Term'] = 360  # Example value, adjust as necessary
    
    # Handle categorical variables
    X.loc[0, 'Credit_History'] = 1 if input_data['Credit_history'] == 'Yes' else 0
    X.loc[0, 'Gender_Female'] = 1 if input_data.get('Gender') == 'Female' else 0
    X.loc[0, 'Gender_Male'] = 1 if input_data.get('Gender') == 'Male' else 0
    X.loc[0, 'Married_No'] = 1 if input_data.get('Married') == 'No' else 0
    X.loc[0, 'Married_Yes'] = 1 if input_data.get('Married') == 'Yes' else 0
    X.loc[0, 'Dependents_0'] = 1 if input_data.get('Dependents') == '0' else 0
    X.loc[0, 'Dependents_1'] = 1 if input_data.get('Dependents') == '1' else 0
    X.loc[0, 'Dependents_2'] = 1 if input_data.get('Dependents') == '2' else 0
    X.loc[0, 'Dependents_3+'] = 1 if input_data.get('Dependents') == '3+' else 0
    X.loc[0, 'Education_Graduate'] = 1 if input_data.get('College_degree') == 'Yes' else 0
    X.loc[0, 'Education_Not Graduate'] = 1 if input_data.get('College_degree') == 'No' else 0
    X.loc[0, 'Self_Employed_No'] = 1 if input_data.get('Self_Employed') == 'No' else 0
    X.loc[0, 'Self_Employed_Yes'] = 1 if input_data.get('Self_Employed') == 'Yes' else 0
    X.loc[0, 'Property_Area_Rural'] = 1 if input_data.get('Community_type') == 'Rural' else 0
    X.loc[0, 'Property_Area_Semiurban'] = 1 if input_data.get('Community_type') == 'Suburban' else 0
    X.loc[0, 'Property_Area_Urban'] = 1 if input_data.get('Community_type') == 'Urban' else 0
    
    # Ensure all columns are present and in correct order
    X = X.astype(float)  # Convert DataFrame to float dtype
    
    # Add columns that were not explicitly set, defaulting to NaN
    missing_cols = set(choices.keys()) - set(X.columns)
    for col in missing_cols:
        X.loc[0, col] = np.nan
    
    # Debugging statements to inspect the data
    print("Input DataFrame:")
    print(X.head())
    
    # Scale input data
    X_transformed = scaler.transform(X)
    print("Transformed Features:")
    print(X_transformed)
    
    # Make prediction
    output = model.predict(X_transformed)
    print("Model Output:")
    print(output)
    
    # Adjusted thresholding logic to determine binary class
    prediction = (output[0] < 0.5).astype(int)
    
    # Return 'No' for 0 and 'Yes' for 1
    return 'No' if prediction == 0 else 'Yes'

@app.route("/", methods=['GET', 'POST'])
def index():
    """
    Function to handle GET and POST requests. Returns result only if POST request.
    """
    if request.method == 'POST':
        # Extract and process input data
        input_data = {
            'Combined_income': request.form.get('Combined_income', type=float),
            'Credit_history': request.form.get('Credit_history'),
            'Requested_amount': request.form.get('Requested_amount', type=float),
            'Community_type': request.form.get('Community_type'),
            'Dependents': request.form.get('Dependents'),
            'College_degree': request.form.get('College_degree'),
            'Gender': request.form.get('Gender'),
            'Married': request.form.get('Married'),
            'Self_Employed': request.form.get('Self_Employed')
        }
        
        result = predict(input_data)
        return render_template('index.html', result=result)
    
    return render_template('index.html', result=None)

@app.route("/tableau")
def tableau():
    """
    Route to render the Tableau dashboard
    """
    return render_template('tableau.html')

if __name__ == '__main__':
    app.run(debug=True)




