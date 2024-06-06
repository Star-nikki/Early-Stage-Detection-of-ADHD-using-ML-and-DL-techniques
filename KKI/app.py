from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load the SVM model
with open('svm_model.pkl', 'rb') as model_file:
    svm_model = pickle.load(model_file)

# Load the CSV file
df = pd.read_csv("KKI_phenotypic_1.csv")

# Define the columns used in the model
input_columns = ['Gender', 'Age', 'Handedness','ADHD Measure','ADHD Index','Inattentive','Hyper/Impulsive','Verbal IQ','Performance IQ','Med Status']

@app.route('/')
def index():
    return render_template('index.html', columns=input_columns)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from the form
        input_values = [float(request.form[column]) for column in input_columns]
        
        # Create a DataFrame with the input values
        input_data = pd.DataFrame([input_values], columns=input_columns)

        # Debugging: Print the input data
        print("Input Data:")
        print(input_data)

        # Get decision function output
        decision_function_output = svm_model.decision_function(input_data)

        # Get the predicted class label
        predicted_class_label = svm_model.classes_[decision_function_output.argmax(axis=1)]

        # Print the decision function output
        print("Decision Function Output:")
        print(decision_function_output)

        return render_template('result.html', prediction=predicted_class_label)
    except Exception as e:
        return render_template('error.html', error=str(e))
   
if __name__ == '__main__':
    app.run(debug=True)

# from flask import Flask, render_template, request, jsonify
# import pickle
# import numpy as np
# import logging

# # Configure logging
# logging.basicConfig(level=logging.DEBUG)

# app = Flask(__name__)

# # Load the trained model
# model = pickle.load(open('knn_model.pkl', 'rb'))

# # Define the features
# FEATURES = ['ScanDir ID', 'Site', 'Gender', 'Age', 'Handedness', 'ADHD Measure', 'ADHD Index', 'Inattentive',
#        'Hyper/Impulsive', 'IQ Measure', 'Verbal IQ', 'Performance IQ', 'Full4 IQ', 'Med Status', 'QC_Rest_1', 'QC_Anatomical_1'
#        ]  # list the features as per your code
# TARGET = 'DX'

# @app.route('/')
# def home():
#     return render_template('index.html', features=FEATURES)

# # @app.route('/', methods=['POST'])
# # def predict():
# #     data = request.form.to_dict()
# #     values = np.array([float(data[feature]) for feature in FEATURES]).reshape(1, -1)
# #     prediction = model.predict(values)
# #     return render_template('result.html', prediction=int(prediction[0]))

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         data = request.form.to_dict()
#         logging.debug(f"Form data received: {data}")
        
#         values = np.array([float(data[feature]) for feature in FEATURES]).reshape(1, -1)
#         logging.debug(f"Values for prediction: {values}")
        
#         prediction = model.predict(values)
#         logging.debug(f"Prediction result: {prediction}")
        
#         return render_template('result.html', prediction=int(prediction[0]))
#     except Exception as e:
#         logging.error(f"Error during prediction: {e}")
#         return render_template('error.html', message=str(e))
    
# if __name__ == '__main__':
#     app.run(debug=True)
