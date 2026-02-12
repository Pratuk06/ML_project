import streamlit as st
import pandas as pd
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

# --- 1. Load Model and Preprocessor ---
try:
    with open('rf_custom_model.pkl', 'rb') as file:
        rf_custom_model = pickle.load(file)
    with open('custom_preprocessor.pkl', 'rb') as file:
        custom_preprocessor = pickle.load(file)
except FileNotFoundError:
    st.error("Model or preprocessor file not found. Make sure 'rf_custom_model.pkl' and 'custom_preprocessor.pkl' are in the same directory.")
    st.stop()

# --- 2. Load original data for unique categorical values ---
# This is needed to populate the select boxes for brand and model
# Assuming the dataset is available in the same directory as app.py
try:
    df_original = pd.read_csv('mobile_resale_dataset_custom.csv')
    unique_brands = df_original['brand'].unique().tolist()
    unique_models = df_original['model'].unique().tolist()
except FileNotFoundError:
    st.error("Original dataset 'mobile_resale_dataset_custom.csv' not found. Cannot populate brand/model options.")
    # Provide dummy lists if the file is not found to allow the app to run
    unique_brands = ["Brand A", "Brand B", "Brand C"]
    unique_models = ["Model X", "Model Y", "Model Z"]
except Exception as e:
    st.error(f"Error loading original dataset: {e}")
    unique_brands = ["Brand A", "Brand B", "Brand C"]
    unique_models = ["Model X", "Model Y", "Model Z"]

# Sort for better user experience
unique_brands.sort()
unique_models.sort()

# --- 3. Streamlit App Interface ---
st.title('Mobile Phone Resale Price Predictor')
st.write('Enter the details of the mobile phone to get an estimated resale price.')

# Input widgets
selected_brand = st.selectbox('Brand', unique_brands)
selected_model = st.selectbox('Model', unique_models)

age_months = st.slider('Age of Phone (months)', min_value=0, max_value=60, value=12)
ram_gb = st.selectbox('RAM (GB)', [4, 6, 8, 12, 16])
storage_gb = st.selectbox('Storage (GB)', [64, 128, 256, 512, 1024])
camera_mp = st.slider('Camera (MP)', min_value=8, max_value=200, value=48)

# Prediction button
if st.button('Predict Resale Price'):
    # Create a DataFrame from inputs
    input_data = pd.DataFrame([{
        'brand': selected_brand,
        'model': selected_model,
        'age_months': age_months,
        'ram_gb': ram_gb,
        'storage_gb': storage_gb,
        'camera_mp': camera_mp
    }])

    try:
        # Preprocess the input data
        input_preprocessed = custom_preprocessor.transform(input_data)

        # Make prediction
        prediction = rf_custom_model.predict(input_preprocessed)[0]

        # Display result
        st.success(f'Estimated Resale Price: ₹{prediction:,.2f}')
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.warning("Please ensure all inputs are valid and try again.")
