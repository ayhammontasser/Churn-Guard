
# import numpy as np
# from flask import Flask, render_template, request,redirect, url_for, flash
# import pickle
# import os
# # Create the application object
# app = Flask(__name__)

# # Load the model
# try:
#     model = pickle.load(open('stack_model.pkl', 'rb'))
# except Exception as e:
#     print(f"Error loading the model: {e}")

# # Load the scaler
# try:
#     scaler = pickle.load(open('scaler.pkl', 'rb'))
# except Exception as e:
#     print(f"Error loading the scaler: {e}")

# @app.route('/')
# def home():
#     return render_template('index.html')


# @app.route("/predict", methods=["POST"])
# def predict():
#     features = [float(x) for x in request.form.values()]
#     features_arr = np.array([features])
#     # Scale the features
#     features_scaled = scaler.transform(features_arr)
#     # Make prediction
#     pred = model.predict(features_scaled)[0]

#     #  Convert output
#     prediction = "Canceled (Churn = YES)" if pred == 1 else "Not Canceled (Churn = NO)"


#     return render_template("index.html", prediction=prediction)


# if __name__ == "__main__":
#     app.run(debug=True)



#import numpy as np
# from flask import Flask, render_template, request
# import pickle

# app = Flask(__name__)

# # Load model
# with open('stack_model.pkl', 'rb') as f:
#     model = pickle.load(f)

# # Load scaler
# with open('scaler.pkl', 'rb') as f:
#     scaler = pickle.load(f)


# @app.route('/')
# def home():
#     return render_template('index.html')


# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Read inputs by name (must match HTML)
#         MonthlyCharges = float(request.form['MonthlyCharges'])
#         TotalCharges = float(request.form['TotalCharges'])
#         tenure = float(request.form['tenure'])
#         Contract_encoded = float(request.form['Contract_encoded'])

#         # Feature order must match training
#         features = np.array([[MonthlyCharges, TotalCharges, tenure, Contract_encoded]])

#         # Scale
#         features_scaled = scaler.transform(features)

#         # Predict probability
#         y_prob = model.predict_proba(features_scaled)[0][1]

#         # Custom threshold
#         y_pred = 1 if y_prob > 0.45 else 0

#         prediction_text = (
#             "Customer Will Churn ❌" if y_pred == 1 
#             else "Customer Will NOT Churn ✔"
#         )

#         return render_template(
#             "index.html",
#             prediction=prediction_text,
#             prob=y_prob
#         )

#     except Exception as e:
#         return f"Error: {e}"


# if __name__ == "__main__":
#     app.run(debug=True)

# 

import numpy as np
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)


# Load model
with open('stack_model.pkl', 'rb') as f:
    stack_model = pickle.load(f)

# Load scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Feature order must match training
feature_order = [
    'MonthlyCharges','TotalCharges','tenure','AvgCharges','clv','time_to_churn',
    'is_long_term','SeniorCitizen',
    'Contract_encoded', 'CampaignResponse', 'NumServices',
    'PaymentMethod_Credit card (automatic)','PaymentMethod_Electronic check','PaymentMethod_Mailed check',
    'PhoneService_Yes','MultipleLines_Yes','OnlineSecurity_Yes','OnlineBackup_Yes','DeviceProtection_Yes',
    'TechSupport_Yes','StreamingTV_Yes','StreamingMovies_Yes','PaperlessBilling_Yes',
    'InternetService_Fiber optic','InternetService_No',  
    'LoyaltyClass_Mid','LoyaltyClass_New','LoyaltyClass_nan',
    'RevenueSegment_Low','RevenueSegment_Medium',
    'clv_segment_1','clv_segment_2',
    'gender_Male','Partner_Yes','Dependents_Yes'
]

@app.route('/')
def home():
    return render_template('index.html')


@app.route("/predict", methods=["POST"])
def predict():

    # ⬅️ 1) Read selected services
    selected_services = request.form.getlist("Services[]")  # list of strings

    # ⬅️ 2) Read non-service inputs
    raw_inputs = {}
    for f in feature_order:
        if f not in selected_services and f not in [
            'PhoneService_Yes','MultipleLines_Yes','OnlineSecurity_Yes',
            'OnlineBackup_Yes','DeviceProtection_Yes','TechSupport_Yes',
            'StreamingTV_Yes','StreamingMovies_Yes','PaperlessBilling_Yes',
            'InternetService_Fiber optic','InternetService_No'
        ]:
            value = request.form.get(f)
            raw_inputs[f] = float(value) if value else 0

    # ⬅️ 3) Build final feature list in correct order
    features = []
    for f in feature_order:
        if f in selected_services:
            features.append(1.0)
        elif f in [
            'PhoneService_Yes','MultipleLines_Yes','OnlineSecurity_Yes',
            'OnlineBackup_Yes','DeviceProtection_Yes','TechSupport_Yes',
            'StreamingTV_Yes','StreamingMovies_Yes','PaperlessBilling_Yes',
            'InternetService_Fiber optic','InternetService_No'
        ]:
            features.append(0.0)
        else:
            features.append(raw_inputs.get(f, 0))

    # Convert to array
    features_arr = np.array([features])

    # Scale
    features_scaled = scaler.transform(features_arr)

    # Predict
    pred = stack_model.predict(features_scaled)[0]
    pred_prob = stack_model.predict_proba(features_scaled)[0, 1]

    prediction = "Canceled (Churn = YES)" if pred == 1 else "Not Canceled (Churn = NO)"

    return render_template("index.html", prediction=prediction, prob=pred_prob)


if __name__ == "__main__":
    app.run(debug=True)
