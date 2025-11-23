import streamlit as st
import pickle
import numpy as np

# Load models
ridge_model = pickle.load(open('models/Ridge.pkl','rb'))
St_Sc = pickle.load(open('models/Scaler.pkl','rb'))
ridgeCV_model = pickle.load(open('models/Ridge_cv.pkl','rb'))

# Page config
st.set_page_config(page_title="Fire Weather Index", layout="centered")

st.image("FWI.jpg", caption=None)
# Custom CSS
st.markdown("""
    <style>
        body {
            background-color: black;
        }
        .main {
            background-color: black;
        }
        label, .st-bb, .st-at, .st-c8, .st-ci {
            color: orange !important;
        }
        h1, h2, h3, p {
            color: orange;
        }
        .result-box {
            background-color: #222;
            padding: 15px;
            border-radius: 10px;
            color: orange;
            text-align: center;
            border: 1px solid orange;
            font-size: 22px;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 style='text-align:center;'>Fire Weather Index (FWI)</h1>", unsafe_allow_html=True)

# Subtitle
st.markdown("""
<p style='text-align:center; font-size:17px;'>
a key metric used in Algeria to assess the potential for forest fires, especially in the central and northern coastal regions that have significant forest cover
</p>
""", unsafe_allow_html=True)

st.write("---")

# Input fields
Temperature = st.number_input("Temperature", step=0.1)
RH = st.number_input("Relative Humidity (RH)", step=0.1)
Ws = st.number_input("Wind Speed (Ws)", step=0.1)
Rain = st.number_input("Rain", step=0.1)
FFMC = st.number_input("FFMC", step=0.1)
DMC = st.number_input("DMC", step=0.1)
ISI = st.number_input("ISI", step=0.1)
Classes = st.number_input("Classes", step=0.1)
Region = st.number_input("Region", step=0.1)

if st.button("Predict Fire Weather Index"):
    new_scaled = St_Sc.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
    
    result = ridge_model.predict(new_scaled)[0]
    res = ridgeCV_model.predict(new_scaled)[0]

    st.markdown(f"""
    <div class='result-box'>
        Ridge Prediction: {result:.3f}<br>
        RidgeCV Prediction: {res:.3f}
    </div>
    """, unsafe_allow_html=True)