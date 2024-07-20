from flask import Flask, render_template, request
import pickle
import pandas as pd

# Create the Flask app
app = Flask(__name__)

# Load assets
with open('training/model.pkl', 'rb') as f: 
    model = pickle.load(f)
with open('training/scaler.pkl', 'rb') as f: 
    scaler = pickle.load(f)
with open('training/choices.pkl', 'rb') as f: 
    choices = pickle.load(f)

# Get list of categorical and numerical columns
cat_cols = [col for col in choices.keys() if choices[col] is not None]
cont_cols = [col for col in choices.keys() if choices[col] is None]

@app.route("/", methods=['GET', 'POST'])
def index():
    """
    Function to handle GET and POST requests. Returns result only if POST request.
    """
    if request.method == 'POST':
        input_data = request.form.to_dict()
        result = predict(input_data)
        return render_template('index.html', choices=choices, result=result)
    else: 
        return render_template('index.html', choices=choices)
    
@app.route("/tableau")
def tableau():
    """
    Route to render the Tableau dashboard
    """
    return render_template('tableau.html')

@app.route('/tableau')
def tableau():
    """
    Route to render the Tableau dashboard
    """
    return render_template('tableau.html')

def predict(input_data):
    """
    Helper function to make predictions based on input data.
    """
    # Create input DataFrame with the correct columns
    input_df = pd.DataFrame([input_data])
    X = pd.DataFrame(columns=choices.keys())
    X = pd.concat([X, input_df], ignore_index=True)
    
    # Handle categorical features: Convert to numeric if necessary
    X[cat_cols] = X[cat_cols].astype('category').apply(lambda x: x.cat.codes)
    
    # Handle continuous features: Convert to float
    X[cont_cols] = X[cont_cols].astype(float)
    
    # Ensure all columns are present
    X = X.reindex(columns=choices.keys(), fill_value=0)
    
    # Scale input data
    X_transformed = scaler.transform(X)
    
    # Make prediction
    output = model.predict(X_transformed)
    
    # Return 'No' for 0 and 'Yes' for 1
    return 'Yes' if output[0] == 1 else 'No'

if __name__ == '__main__':
    app.run()