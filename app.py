
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import PolynomialFeatures
import requests

# Load PM10 prediction model and DataFrame
with open('pmmodel.pkl', 'rb') as pmmodel_file:
    pm_model = pickle.load(pmmodel_file)

# Load PM2.5 prediction model
with open('fpmodel.pkl', 'rb') as fpmodel_file:
    fp_model = pickle.load(fpmodel_file)

# Load DataFrame for PM2.5 prediction
with open('df.pkl', 'rb') as df_file:
    df = pickle.load(df_file)

# Function to calculate AQI based on PM10
def calculate_aqi(PM10):
    aqi = 0
    if PM10 <= 50:
        aqi = PM10
    elif PM10 > 50 and PM10 <= 100:
        aqi = PM10
    elif PM10 > 100 and PM10 <= 250:
        aqi = 100 + (PM10 - 100) * (100 / 150)
    elif PM10 > 250 and PM10 <= 350:
        aqi = 200 + (PM10 - 250)
    elif PM10 > 350 and PM10 <= 450:
        aqi = 300 + (PM10 - 350) * (100 / 80)
    else:
        aqi = 400 + (PM10 - 430) * (100 / 80)
    return aqi

# Function to determine AQI category
def get_aqi_category(aqi):
    if aqi <= 50:
        return 'Good'
    elif aqi <= 100:
        return 'Satisfactory'
    elif aqi <= 200:
        return 'Moderate'
    elif aqi <= 300:
        return 'Poor'
    elif aqi <= 400:
        return 'Very Poor'
    else:
        return 'Severe'

# Function to suggest precautions and mitigation based on AQI category
def suggest_precautions_and_mitigation(aqi_category):
    precautions = ""
    mitigation_measures = ""
    icon = ""

    if aqi_category == 'Good':
        precautions = "No health implications. Enjoy outdoor activities!"
        mitigation_measures = "Continue routine maintenance of air quality management systems."
        icon = "🌞"

    elif aqi_category == 'Satisfactory':
        precautions = "Minor health issues may occur for sensitive individuals. Limit outdoor activities if experiencing symptoms."
        mitigation_measures = "Enhance monitoring and maintenance of air quality management systems."
        icon = "😊"

    elif aqi_category == 'Moderate':
        precautions = "Health effects may be noticeable for sensitive people. Limit prolonged outdoor exertion."
        mitigation_measures = "Implement additional pollution control measures and public awareness campaigns."
        icon = "😐"

    elif aqi_category == 'Poor':
        precautions = "Health effects are likely for everyone. Avoid outdoor activities and stay indoors."
        mitigation_measures = "Implement stringent pollution control measures, reduce traffic, and promote public transport."
        icon = "😷"

    elif aqi_category == 'Very Poor':
        precautions = "Serious health effects for everyone. Avoid outdoor activities completely."
        mitigation_measures = "Implement emergency measures, reduce industrial emissions, and enforce strict vehicle emissions standards."
        icon = "🤢"

    elif aqi_category == 'Severe':
        precautions = "Health emergency. Stay indoors and use air purifiers if available."
        mitigation_measures = "Declare a state of emergency, shut down high-polluting industries, and enforce vehicle use restrictions."
        icon = "🚨"

    return precautions, mitigation_measures, icon

# Function to get color code based on AQI category
def get_aqi_category_color(aqi_category):
    color_mapping = {
        'Good': 'green',
        'Satisfactory': 'lightgreen',
        'Moderate': 'yellow',
        'Poor': 'orange',
        'Very Poor': 'red',
        'Severe': 'brown'
    }
    return color_mapping.get(aqi_category, 'black')

# Real-time data integration
def get_real_time_aqi(api_key, city):
    base_url = "http://api.openweathermap.org/data/2.5/air_pollution"
    complete_url = f"{base_url}?q={city}&appid={api_key}"
    response = requests.get(complete_url)
    data = response.json()

    if "list" in data:
        pm10 = data["list"][0]["components"]["pm10"]
        pm25 = data["list"][0]["components"]["pm2_5"]
        aqi = data["list"][0]["main"]["aqi"]
        return pm10, pm25, aqi
    else:
        st.error("City Not Found")
        return None, None, None

# Streamlit App Configuration
st.set_page_config(page_title="AirWise", page_icon="🌍", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for dark background and white tabs
st.markdown(
    """
    <style>
    .main {
        background-color: #2e2e2e;
        color: #fff;
    }
    .sidebar .sidebar-content {
        background-color: #fff;
        color: #333;
    }
    h1, h2, h3, h4 {
        color: #fff;
    }
    .stButton>button {
        background-color: #2c3e50;
        color: white;
    }
    .stSlider>.ui.range {
        background-color: #2c3e50;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("AirWise")

# Real-time AQI data input
st.sidebar.header("REAL-TIME AQI:")
api_key = st.sidebar.text_input("Enter your API key:")
city = st.sidebar.text_input("Enter your city:")

if st.sidebar.button("Get Real-Time AQI"):
    if api_key and city:
        rt_pm10, rt_pm25, rt_aqi = get_real_time_aqi(api_key, city)
        if rt_pm10 and rt_pm25 and rt_aqi:
            rt_aqi_category = get_aqi_category(rt_aqi)
            rt_color = get_aqi_category_color(rt_aqi_category)
            rt_precautions, rt_mitigation_measures, rt_icon = suggest_precautions_and_mitigation(rt_aqi_category)

            st.subheader("Real-Time AQI Data:")
            st.write(f"**City:** {city}")
            st.write(f"**PM10:** {rt_pm10} µg/m³")
            st.write(f"**PM2.5:** {rt_pm25} µg/m³")
            st.write(f"**AQI:** {rt_aqi}")
            st.markdown(f"<span style='color:{rt_color}; font-weight:bold'>AQI Category: {rt_aqi_category}</span>", unsafe_allow_html=True)

            st.subheader("Real-Time Precautions:")
            st.write(f"<span style='font-size: 24px;'>{rt_icon} {rt_precautions}</span>", unsafe_allow_html=True)

            st.subheader("Real-Time Mitigation Measures:")
            st.write(f"<span style='font-size: 24px;'>{rt_icon} {rt_mitigation_measures}</span>", unsafe_allow_html=True)
    else:
        st.sidebar.error("Please enter both API key and city name.")

# User input for PM10 prediction
st.sidebar.header("INPUT FEATURES:")
input_rf = st.sidebar.slider("Rainfall (mm/day):", 0.0, 500.0, step=0.1)
input_rh = st.sidebar.slider("Relative Humidity (%):", 0.0, 100.0, step=0.1)
input_ws = st.sidebar.slider("Wind Speed (km/hr):", 0.0, 100.0, step=0.1)
input_at = st.sidebar.slider("Average Daily Temperature (degC):", 0.0, 50.0, step=0.1)
input_wd = st.sidebar.slider("Wind Direction (Degrees):", 0.0, 360.0, step=1.0)

# Prepare input data for prediction
input_data = np.array([[input_rf, input_rh, input_ws, input_at, input_wd]])
input_poly = PolynomialFeatures(degree=2).fit_transform(input_data)

# Make prediction for PM10
predicted_pm10 = pm_model.predict(input_poly)[0]

# Calculate AQI based on predicted PM10
predicted_aqi = calculate_aqi(predicted_pm10)

# Predict PM2.5 value based on predicted PM10
pm25_input = np.array([[predicted_pm10]])
predicted_pm25 = fp_model.predict(pm25_input)[0]

aqi_category = get_aqi_category(predicted_aqi)
color = get_aqi_category_color(aqi_category)
precautions, mitigation_measures, icon = suggest_precautions_and_mitigation(aqi_category)

# Display predicted values
st.subheader("Predicted PM10 Value:")
st.write(f"<span style='font-size: 24px;'>{round(predicted_pm10, 1)}</span>", unsafe_allow_html=True)

st.subheader("Predicted PM2.5 Value:")
st.write(f"<span style='font-size: 24px;'>{round(predicted_pm25, 1)}</span>", unsafe_allow_html=True)

st.subheader("Predicted AQI:")
st.write(f"<span style='font-size: 24px; color:{color};'>{round(predicted_aqi, 1)}</span>", unsafe_allow_html=True)

st.subheader(" ")
st.markdown(f"<span style='color:{color}; font-weight:bold'>AQI Category: {aqi_category}</span>", unsafe_allow_html=True)

st.subheader("Precautions:")
st.write(f"<span style='font-size: 24px;'>{icon} {precautions}</span>", unsafe_allow_html=True)

st.subheader("Mitigation Measures:")
st.write(f"<span style='font-size: 24px;'>{icon} {mitigation_measures}</span>", unsafe_allow_html=True)
