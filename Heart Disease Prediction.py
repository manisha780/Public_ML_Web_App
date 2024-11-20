import pickle
import pandas as pd
import streamlit as st

# Load the trained Logistic Regression model and scaler
with open("C:/Users/hp/Downloads/heart_disease_lr_model.pkl", 'rb') as f:
    model = pickle.load(f)

with open("C:/Users/hp/Downloads/scaler (1).pkl", 'rb') as f:
    scaler = pickle.load(f)

# Streamlit App
st.title("Heart Disease Prediction System")

# Sidebar for user input
st.sidebar.header("Input Patient Data")

# Input fields for the user
age = st.sidebar.number_input("Age", min_value=29, max_value=77, value=50)
sex = st.sidebar.selectbox("Sex", options=[0, 1], index=1)
cp = st.sidebar.selectbox("Chest Pain Type", options=[0, 1, 2, 3], index=0)  # Example categories
trestbps = st.sidebar.number_input("Resting Blood Pressure (mm Hg)", min_value=90, max_value=200, value=120)
chol = st.sidebar.number_input("Serum Cholestoral in mg/dl", min_value=100, max_value=600, value=200)
fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1], index=0)
restecg = st.sidebar.selectbox("Resting Electrocardiographic Results", options=[0, 1, 2], index=0)
thalach = st.sidebar.number_input("Maximum Heart Rate Achieved", min_value=70, max_value=220, value=150)
exang = st.sidebar.selectbox("Exercise Induced Angina", options=[0, 1], index=0)
oldpeak = st.sidebar.number_input("Depression Induced by Exercise", min_value=0.0, max_value=6.0, value=1.0)
slope = st.sidebar.selectbox("Slope of the Peak Exercise ST Segment", options=[0, 1, 2], index=0)
ca = st.sidebar.selectbox("Number of Major Vessels Colored by Fluoroscopy", options=[0, 1, 2, 3], index=0)
thal = st.sidebar.selectbox("Thalassemia", options=[1, 2, 3], index=0)

# Collect the input data
input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]],
                          columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang',
                                   'oldpeak', 'slope', 'ca', 'thal'])

# Display input data
st.write("### Input Data")
st.write(input_data)

# Ensure the input data matches the expected format and scale it
scaled_data = scaler.transform(input_data)

# Predict button
if st.button("Predict"):
    # Make prediction
    prediction = model.predict(scaled_data)
    prediction_proba = model.predict_proba(scaled_data)[0][1]

    # Display result
    if prediction[0] == 1:
        st.write("### Prediction: Positive for Heart Disease 🛑")
    else:
        st.write("### Prediction: Negative for Heart Disease ✅")

    st.write(f"### Prediction Probability: {prediction_proba * 100:.2f}% chance of heart disease")
