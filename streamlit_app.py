import streamlit as st
import pickle
import numpy as np

# Load the pre-trained model
with open('sakit_jantung.sav', 'rb') as file:
    sakit_jantung = pickle.load(file)

# Load the scaler used during training
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# App title
st.title('Chance of Survival after a Heart Attack (1 for Alive / 0 for Deceased)')

# Input fields
age = st.number_input('Your Age:', min_value=0, max_value=120, step=1)
anaemia = st.selectbox('Do you have anemia?', [0, 1])
creatinine_phosphokinase = st.number_input('Creatine Phosphokinase Level:', min_value=0)
diabetes = st.selectbox('Do you have diabetes?', [0, 1])
ejection_fraction = st.number_input('Ejection Fraction (% of blood leaving the heart each time it contracts):', min_value=0, max_value=100)
high_blood_pressure = st.selectbox('Do you have high blood pressure?', [0, 1])
platelets = st.number_input('Platelets in blood (kiloplatelets/ml):', min_value=0)
serum_creatinine = st.number_input('Serum Creatinine Level (mg/dl):', min_value=0.0, format="%.2f")
serum_sodium = st.number_input('Serum Sodium Level (mEq/L):', min_value=0)
sex = st.selectbox('Gender (1=Male, 0=Female):', [0, 1])
smoking = st.selectbox('Are you a smoker?', [0, 1])
time = st.number_input('Days since last check-up:', min_value=0)

# Diagnosis result placeholder
jantung_diagnosis = ''

# Prediction
if st.button('Predict Survival'):
    # Convert inputs to the appropriate numerical types
    input_data = np.array([[
        float(age), float(anaemia), float(creatinine_phosphokinase), float(diabetes),
        float(ejection_fraction), float(high_blood_pressure), float(platelets),
        float(serum_creatinine), float(serum_sodium), float(sex), float(smoking), float(time)
    ]])

    # Standardize the input data
    std_data = scaler.transform(input_data)
    
    # Make prediction
    prediction = sakit_jantung.predict(std_data)

    if prediction[0][0] > 0.5:
        jantung_diagnosis = 'The patient has died.'
    else:
        jantung_diagnosis = 'The patient is alive.'
    st.success(jantung_diagnosis)