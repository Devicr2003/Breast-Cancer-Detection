import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the trained model
model_path = '/home/rajani/Desktop/Project/model.pkl'  # path to the uploaded model file
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

# Streamlit app UI
st.title("Breast Cancer Detection App")
st.write("This web app will predict whether a tumor is malignant or benign based on input features.")

# Input features
col1, col2 , col3= st.columns(3)
with col1:
    radius = st.number_input("Radius", min_value=0.0, step=0.1)
    texture = st.number_input("Texture", min_value=0.0, step=0.1)
    perimeter = st.number_input("Perimeter", min_value=0.0, step=0.1)
    area = st.number_input("Area", min_value=0.0, step=0.1)
    smoothness = st.number_input("Smoothness", min_value=0.0, step=0.01)
    compactness = st.number_input("Compactness", min_value=0.0, step=0.01)
    concavity = st.number_input("Concavity", min_value=0.0, step=0.01)
    concave_points = st.number_input("Concave Points", min_value=0.0, step=0.01)
    symmetry = st.number_input("Symmetry", min_value=0.0, step=0.01)
    fractal_dimension = st.number_input("Fractal Dimension", min_value=0.0, step=0.01)

with col2:
    radius_se = st.number_input("Radius SE", min_value=0.0, step=0.1)
    texture_se = st.number_input("Texture SE", min_value=0.0, step=0.1)
    perimeter_se = st.number_input("Perimeter SE", min_value=0.0, step=0.1)
    area_se = st.number_input("Area SE", min_value=0.0, step=0.1)
    smoothness_se = st.number_input("Smoothness SE", min_value=0.0, step=0.01)
    compactness_se = st.number_input("Compactness SE", min_value=0.0, step=0.01)
    concavity_se = st.number_input("Concavity SE", min_value=0.0, step=0.01)
    concave_points_se = st.number_input("Concave Points SE", min_value=0.0, step=0.01)
    symmetry_se = st.number_input("Symmetry SE", min_value=0.0, step=0.01)
    fractal_dimension_se = st.number_input("Fractal Dimension SE", min_value=0.0, step=0.01)

with col3:
    radius_worst = st.number_input("Radius Worst", min_value=0.0, step=0.1)
    texture_worst = st.number_input("Texture Worst", min_value=0.0, step=0.1)
    perimeter_worst = st.number_input("Perimeter Worst", min_value=0.0, step=0.1)
    area_worst = st.number_input("Area Worst", min_value=0.0, step=0.1)
    smoothness_worst = st.number_input("Smoothness Worst", min_value=0.0, step=0.01)
    compactness_worst = st.number_input("Compactness Worst", min_value=0.0, step=0.01)
    concavity_worst = st.number_input("Concavity Worst", min_value=0.0, step=0.01)
    concave_points_worst = st.number_input("Concave Points Worst", min_value=0.0, step=0.01)
    symmetry_worst = st.number_input("Symmetry Worst", min_value=0.0, step=0.01)
    fractal_dimension_worst = st.number_input("Fractal Dimension Worst", min_value=0.0, step=0.01)


  

# Create feature array from user input
features = np.array([[
    radius, texture, perimeter, area, smoothness, compactness, concavity, concave_points, symmetry, fractal_dimension,
    radius_se, texture_se, perimeter_se, area_se, smoothness_se, compactness_se, concavity_se, concave_points_se, symmetry_se, fractal_dimension_se,
    radius_worst, texture_worst, perimeter_worst, area_worst, smoothness_worst, compactness_worst, concavity_worst, concave_points_worst, symmetry_worst, fractal_dimension_worst
]])
# Predict button
if st.button("Predict"):
    # Make prediction
    prediction = model.predict(features)
    
    # Display prediction result
    if prediction == 1:
        st.write("The tumor is **Malignant**.")
    else:
        st.write("The tumor is **Benign**.")
