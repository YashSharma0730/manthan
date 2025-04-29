
import streamlit as st
import numpy as np
import tensorflow as tf
import joblib

model = tf.keras.models.load_model("model.h5")
scaler = joblib.load("scaler.save")

st.title("Rice Yield Prediction App")

soil_moisture = st.number_input("Soil Moisture", value=0.0)
nitrogen = st.number_input("Nitrogen Level", value=0.0)
phosphorus = st.number_input("Phosphorus Level", value=0.0)
potassium = st.number_input("Potassium Level", value=0.0)
soil_ph = st.number_input("Soil pH", value=0.0)
temperature = st.number_input("Temperature (°C)", value=0.0)
rainfall = st.number_input("Rainfall (mm)", value=0.0)
humidity = st.number_input("Humidity (%)", value=0.0)
area = st.number_input("Area (in hectares)", value=0.0)
prev_yield = st.number_input("Previous Yield (kg/ha)", value=0.0)

if st.button("Predict Yield"):
    input_data = np.array([[soil_moisture, nitrogen, phosphorus, potassium, soil_ph,
                            temperature, rainfall, humidity, area, prev_yield]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    st.success(f"Predicted Rice Yield: {prediction[0][0]:.2f} kg/ha")
