
import streamlit as st
import pandas as pd
import joblib

# Load the trained model and feature columns
model = joblib.load('model.pkl')
feature_columns = joblib.load('feature_columns.pkl')

# Load unique values for dropdowns
unique_ram = joblib.load('unique_ram.pkl')
unique_storage = joblib.load('unique_storage.pkl')
unique_camera = joblib.load('unique_camera.pkl')

st.title('Resale Price Prediction App')
st.write('Enter the details of the phone to predict its resale price.')

# Input widgets for the main features
age_months = st.slider('Age (in months)', min_value=0, max_value=60, value=12)
ram_gb = st.selectbox('RAM (GB)', options=unique_ram, index=unique_ram.index(8) if 8 in unique_ram else 0)
storage_gb = st.selectbox('Storage (GB)', options=unique_storage, index=unique_storage.index(128) if 128 in unique_storage else 0)
camera_mp = st.selectbox('Camera (MP)', options=unique_camera, index=unique_camera.index(48) if 48 in unique_camera else 0)

# Placeholder for categorical features. For this simple app, we will assume average/default values.
# In a more complex app, dropdowns for brand/model would be added.
# For now, create a base dataframe with all features initialized to 0, then fill known inputs.
input_data = pd.DataFrame(0, index=[0], columns=feature_columns)

# Fill in the user-provided numerical features
input_data['age_months'] = age_months
input_data['ram_gb'] = ram_gb
input_data['storage_gb'] = storage_gb
input_data['camera_mp'] = camera_mp

# Prediction button
if st.button('Predict Resale Price'):
    # Make prediction
    prediction = model.predict(input_data)[0]
    st.success(f'Predicted Resale Price: ₹{prediction:,.2f}')
