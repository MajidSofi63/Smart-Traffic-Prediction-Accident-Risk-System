import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the trained models
try:
    traffic_speed_model = joblib.load('traffic_speed_model.joblib')
    accident_risk_model = joblib.load('accident_risk_model.joblib')
    st.success('Models loaded successfully!')
except FileNotFoundError:
    st.error('Model files not found. Please ensure `traffic_speed_model.joblib` and `accident_risk_model.joblib` are in the same directory.')
    st.stop()

st.title('Traffic Prediction and Accident Risk System')
st.write('Enter parameters to get traffic speed prediction and accident risk assessment.')

# User input for traffic speed prediction
st.header('Traffic Speed Prediction')
hour_speed = st.slider('Hour (0-23) for Speed', 0, 23, 8)
dayofweek_speed = st.slider('Day of Week (0=Monday, 6=Sunday) for Speed', 0, 6, 0)
month_speed = st.slider('Month (1-12) for Speed', 1, 12, 1)

if st.button('Predict Traffic Speed'):
    speed_input = pd.DataFrame([[hour_speed, dayofweek_speed, month_speed]],
                                columns=['hour', 'dayofweek', 'month'])
    predicted_speed = traffic_speed_model.predict(speed_input)[0]
    st.write(f'Predicted Traffic Speed: {predicted_speed:.2f} kph')
    st.session_state.predicted_speed = predicted_speed # Store for accident prediction

# User input for accident risk prediction
st.header('Accident Risk Prediction')
hour_accident = st.slider('Hour (0-23) for Accident Risk', 0, 23, 8)
dayofweek_accident = st.slider('Day of Week (0=Monday, 6=Sunday) for Accident Risk', 0, 6, 0)
month_accident = st.slider('Month (1-12) for Accident Risk', 1, 12, 1)

# Use predicted speed if available, otherwise ask for input
current_speed_kph = st.number_input('Current Speed (kph)', min_value=0.0, max_value=150.0, value=st.session_state.get('predicted_speed', 60.0))
current_volume = st.number_input('Current Volume', min_value=0, max_value=500, value=100)

if st.button('Predict Accident Risk'):
    accident_input = pd.DataFrame([[current_speed_kph, current_volume, hour_accident, dayofweek_accident, month_accident]],
                                  columns=['Speed_kph', 'Volume', 'hour', 'dayofweek', 'month'])
    accident_prediction = accident_risk_model.predict(accident_input)[0]
    accident_proba = accident_risk_model.predict_proba(accident_input)[0][1]

    st.write(f'Predicted Accident Occurred: {accident_prediction} (0=No, 1=Yes)')
    st.write(f'Probability of Accident: {accident_proba:.2f}')

    if accident_prediction == 1:
        st.error('High Accident Risk!')
    else:
        st.success('Low Accident Risk.')


# Instructions to run the app
st.sidebar.markdown("""
### How to run this app:
1. Save the code above into a file named `app.py`.
2. In your terminal, navigate to the directory where you saved `app.py`.
3. Run the command: `streamlit run app.py`
4. The app will open in your web browser.
""")

print("Streamlit app code generated. Save this to 'app.py' and run with 'streamlit run app.py'")