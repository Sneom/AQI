import pandas as pd
import streamlit as st
import joblib
from datetime import datetime

# Load the trained model
model = joblib.load('aqi_model.pkl')

# Title of the app
st.title("AQI Prediction App")

# Sidebar for user input
st.sidebar.header('User Input Features')

def user_input_features():
    pm25 = st.sidebar.number_input('PM2.5', min_value=0.0, max_value=500.0, value=50.0)
    pm10 = st.sidebar.number_input('PM10', min_value=0.0, max_value=600.0, value=100.0)
    no2 = st.sidebar.number_input('NO2', min_value=0.0, max_value=200.0, value=40.0)
    co = st.sidebar.number_input('CO', min_value=0.0, max_value=10.0, value=0.5)
    so2 = st.sidebar.number_input('SO2', min_value=0.0, max_value=100.0, value=20.0)
    temp = st.sidebar.number_input('Temperature', min_value=-30.0, max_value=50.0, value=20.0)

    now = datetime.now()

    data = {
        'PM2.5': pm25,
        'PM10': pm10,
        'NO2': no2,
        'CO': co,
        'SO2': so2,
        'Temp': temp,
        'Hour': now.hour,
        'Day': now.day,
        'Month': now.month,
        'Year': now.year
        
    }

    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

st.subheader('User Input Features')
st.write(input_df)

if st.button('Predict AQI'):
    try:
        # Define the exact feature names used in training
        expected_features = ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'Temp',  'Day', 'Hour','Year', 'Month']
        
        # Reorder or filter the input DataFrame to match the training data features
        input_df = input_df.reindex(columns=expected_features, fill_value=0)
        
        # Predict AQI using the loaded model
        prediction = model.predict(input_df)
        st.subheader('Predicted AQI')
        st.write(prediction[0])
    except Exception as e:
        st.error(f"Error: {e}")
