# app.py

import streamlit as st
import pickle
import numpy as np

# Load the model
try:
    with open("rent_model.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Model file not found. Please run train_model.py to create rent_model.pkl.")
    st.stop()

# App title
st.title("🏠 Apartment Rent Prediction App")

st.write("Enter apartment details to predict the monthly rent:")

# Input fields
area = st.number_input("Area (sq ft)", min_value=100, max_value=10000, value=1000)
bedrooms = st.slider("Number of Bedrooms", 0, 10, 2)
bathrooms = st.slider("Number of Bathrooms", 0, 10, 1)

# Prediction
if st.button("Predict Rent"):
    features = np.array([[area, bedrooms, bathrooms]])
    prediction = model.predict(features)
    st.success(f"Estimated Monthly Rent: ${prediction[0]:,.2f}")
