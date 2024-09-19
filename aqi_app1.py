import pandas as pd
import numpy as np
import onnxruntime as ort
import streamlit as st
from datetime import datetime

# Load the ONNX model
onnx_model_path = 'aqi_model.onnx'
ort_session = ort.InferenceSession(onnx_model_path)

# Title of the app
st.title("AQI Prediction App")

# Sidebar for user input
st.sidebar.header('User Input Features')

def user_input_features():
    pm25 = st.sidebar.number_input('PM2.5', min_value=0.0, max_value=500.0, value=50.0)
    pm10 = st.sidebar.number_input('PM10', min_value=0.0, max_value=600.0, value=100.0)
    no2 = st.sidebar.number_input('NO2', min_value=0.0, max_value=200.0, value=40.0)
    co = st.sidebar.number_input('CO', min_value=0.0, max_value=10.0, value=0.5)
    temp = st.sidebar.number_input('Temperature', min_value=-30.0, max_value=50.0, value=20.0)

    now = datetime.now()

    # Use the features that match the model's expected input
    data = {
        'PM2.5': pm25,
        'PM10': pm10,
        'NO2': no2,
        'CO': co,
        'Temp': temp,
        'Year': now.year
    }

    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

st.subheader('User Input Features')
st.write(input_df)

if st.button('Predict AQI'):
    try:
        # Ensure the input DataFrame has the same columns as the model expects
        expected_features = ['PM2.5', 'PM10', 'NO2', 'CO', 'Temp', 'Year']
        input_df = input_df.reindex(columns=expected_features, fill_value=0)
        
        # Convert input DataFrame to numpy array
        input_array = input_df.to_numpy(dtype=np.float32)
        
        # Perform prediction
        ort_inputs = {ort_session.get_inputs()[0].name: input_array}
        prediction = ort_session.run(None, ort_inputs)[0]
        
        st.subheader('Predicted AQI')
        st.write(prediction[0])
    except Exception as e:
        st.error(f"Error: {e}")
