import streamlit as st
import pandas as pd
import joblib

st.title("Energy Consumption Classifier")

# Load model and encoders with error handling
try:
    model = joblib.load("naive_bayes_energy_model.pkl")
    label_encoders = joblib.load("label_encoders.pkl")
    st.success("Model and encoders loaded successfully!")
except Exception as e:
    st.error(f"Error loading model or encoders: {e}")
    st.stop()

# Input form
try:
    temperature = st.slider("Temperature", 0.0, 50.0, 25.0)
    humidity = st.slider("Humidity", 0.0, 100.0, 50.0)
    square_footage = st.slider("Square Footage", 100.0, 10000.0, 1000.0)
    occupancy = st.slider("Occupancy", 0, 500, 50)
    hvac = st.selectbox("HVAC Usage", ['Off', 'On'])
    lighting = st.selectbox("Lighting Usage", ['Off', 'On'])
    renewable = st.slider("Renewable Energy", 0.0, 100.0, 20.0)
    day = st.selectbox("Day of Week", label_encoders['DayOfWeek'].classes_)
    holiday = st.selectbox("Holiday", ['No', 'Yes'])

    input_df = pd.DataFrame([{
        "Temperature": temperature,
        "Humidity": humidity,
        "SquareFootage": square_footage,
        "Occupancy": occupancy,
        "HVACUsage": label_encoders['HVACUsage'].transform([hvac])[0],
        "LightingUsage": label_encoders['LightingUsage'].transform([lighting])[0],
        "RenewableEnergy": renewable,
        "DayOfWeek": label_encoders['DayOfWeek'].transform([day])[0],
        "Holiday": label_encoders['Holiday'].transform([holiday])[0]
    }])

    if st.button("Predict"):
        prediction = model.predict(input_df)[0]
        result = label_encoders['EnergyClass'].inverse_transform([prediction])[0]
        st.success(f"Predicted Energy Consumption: **{result}**")

except Exception as e:
    st.error(f"An error occurred during input or prediction: {e}")
