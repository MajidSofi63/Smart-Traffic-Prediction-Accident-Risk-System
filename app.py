from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import requests
import os
import sib_api_v3_sdk
from datetime import datetime

app = Flask(__name__)
app.secret_key = "traffic_prediction_secret_key_2024"

# Load your trained model
model = None
label_encoders = None
target_encoder = None
features = None

def load_models():
    global model, label_encoders, target_encoder, features
    try:
        model = joblib.load('model/traffic_model.pkl')
        label_encoders = joblib.load('model/label_encoders.pkl')
        target_encoder = joblib.load('model/target_encoder.pkl')
        features = joblib.load('model/selected_features.pkl')
        print("✅ Model loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Model loading error: {e}")
        return False

# Load models at startup
load_models()

# TomTom API Key (Optional - set in Render env)
TOMTOM_API_KEY = os.environ.get('TOMTOM_API_KEY', '')

# Brevo Email Config
BREVO_API_KEY = os.environ.get('BREVO_API_KEY', '')
BREVO_SENDER_EMAIL = os.environ.get('BREVO_SENDER_EMAIL', 'majidsofi63@gmail.com')

def send_email_alert(severity, confidence, risk_factors, user_email, risk_level):
    if not user_email or not BREVO_API_KEY:
        return False
    try:
        configuration = sib_api_v3_sdk.Configuration()
        configuration.api_key['api-key'] = BREVO_API_KEY
        api_instance = sib_api_v3_sdk.TransactionalEmailsApi(sib_api_v3_sdk.ApiClient(configuration))
        
        subject = f"Trip Risk Assessment: {severity} Accident Risk - {risk_level}"
        
        html = f"""
        <html><body>
        <h2>🚗 Smart Traffic Prediction System</h2>
        <h3 style="color: {'red' if risk_level=='HIGH' else 'orange' if risk_level=='MEDIUM' else 'green'}">
            Risk Level: {risk_level}
        </h3>
        <p><strong>Predicted Severity:</strong> {severity}</p>
        <p><strong>Confidence:</strong> {confidence}%</p>
        <h3>⚠️ Risk Factors:</h3><ul>
        """
        for factor in risk_factors[:5]:
            html += f"<li>{factor}</li>"
        
        html += "</ul><p>Stay safe! 🚗</p></body></html>"
        
        sender = {"name": "Smart Traffic System", "email": BREVO_SENDER_EMAIL}
        to = [{"email": user_email}]
        
        send_smtp_email = sib_api_v3_sdk.SendSmtpEmail(
            to=to, sender=sender, subject=subject, html_content=html
        )
        
        api_instance.send_transac_email(send_smtp_email)
        return True
    except Exception as e:
        print(f"Email error: {e}")
        return False

def geocode_address(address):
    """Convert address to coordinates using Nominatim (free, no API key)"""
    url = f"https://nominatim.openstreetmap.org/search?format=json&q={address}&limit=1"
    try:
        response = requests.get(url, headers={'User-Agent': 'TrafficApp/1.0'}, timeout=10)
        data = response.json()
        if data:
            return {'lat': float(data[0]['lat']), 'lon': float(data[0]['lon'])}
    except:
        pass
    return None

def get_route_osrm(start_lat, start_lon, end_lat, end_lon):
    """Get route using OSRM (free, no API key)"""
    url = f"https://router.project-osrm.org/route/v1/driving/{start_lon},{start_lat};{end_lon},{end_lat}?overview=full&geometries=geojson"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        if data.get('code') == 'Ok':
            route = data['routes'][0]
            leg = route['legs'][0]
            points = []
            for coord in route['geometry']['coordinates']:
                points.append([coord[1], coord[0]])
            return {
                'distance_km': route['distance'] / 1000,
                'duration_min': route['duration'] / 60,
                'points': points
            }
    except Exception as e:
        print(f"OSRM error: {e}")
    return None

def predict_risk_with_model(distance_km, duration_min, vehicle_type, time_of_day, weather):
    """Use your trained model to predict risk"""
    
    # Map inputs to model features
    day_map = {'Morning': 'Monday', 'Afternoon': 'Tuesday', 'Evening': 'Friday', 'Night': 'Saturday'}
    light_map = {'Morning': 'Daylight', 'Afternoon': 'Daylight', 'Evening': 'Darkness - lights lit', 'Night': 'Darkness - no lighting'}
    weather_map = {'Clear': 'Fine no high winds', 'Rain': 'Raining no high winds', 'Fog': 'Fog or mist', 'Snow': 'Snowing no high winds'}
    
    # Calculate speed
    avg_speed_kmh = distance_km / (duration_min / 60) if duration_min > 0 else 50
    speed_limit_mph = min(int(avg_speed_kmh * 0.62), 70)
    
    input_data = {
        'Day_of_Week': day_map.get(time_of_day, 'Monday'),
        'Junction_Control': 'Not at junction or within 20m',
        'Light_Conditions': light_map.get(time_of_day, 'Daylight'),
        'Road_Surface_Conditions': 'Wet or damp' if weather != 'Clear' else 'Dry',
        'Road_Type': 'Single carriageway',
        'Speed_limit': speed_limit_mph,
        'Urban_or_Rural_Area': 'Rural' if distance_km > 100 else 'Urban',
        'Weather_Conditions': weather_map.get(weather, 'Fine no high winds'),
        'Number_of_Vehicles': 2,
        'Number_of_Casualties': 1
    }
    
    # Create dataframe and encode
    input_df = pd.DataFrame([input_data])
    for column in input_df.columns:
        if column in label_encoders:
            le = label_encoders[column]
            try:
                input_df[column] = le.transform(input_df[column].astype(str))
            except:
                input_df[column] = 0
    
    # Predict
    prediction = model.predict(input_df)
    severity = target_encoder.inverse_transform(prediction)[0]
    probabilities = model.predict_proba(input_df)[0]
    confidence = max(probabilities) * 100
    
    # Determine risk level
    if severity == 'Fatal' or severity == 'Serious':
        risk_level = 'HIGH'
    else:
        risk_level = 'MEDIUM' if confidence > 70 else 'LOW'
    
    # Generate risk factors
    risk_factors = [
        f"Distance: {distance_km:.0f} km",
        f"Expected duration: {duration_min:.0f} minutes",
        f"Travel time: {time_of_day}",
        f"Weather: {weather}"
    ]
    
    if vehicle_type == 'Motorcycle':
        risk_factors.append("Motorcycle - Higher vulnerability")
    if distance_km > 200:
        risk_factors.append("Long distance - Fatigue risk")
    if time_of_day == 'Night':
        risk_factors.append("Night travel - Reduced visibility")
    
    return {
        'risk_level': risk_level,
        'severity': severity,
        'confidence': round(confidence, 2),
        'risk_factors': risk_factors
    }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/map')
def map_route():
    return render_template('map_route.html')

@app.route('/calculate_route', methods=['POST'])
def calculate_route():
    try:
        data = request.json
        print(f"Received request: {data}")
        
        start = data.get('start')
        end = data.get('end')
        vehicle_type = data.get('vehicle_type', 'Car')
        time_of_day = data.get('time_of_day', 'Morning')
        weather = data.get('weather', 'Clear')
        user_email = data.get('user_email', '')
        
        if not start or not end:
            return jsonify({'error': 'Missing start or destination'}), 400
        
        # Geocode addresses
        start_coords = geocode_address(start)
        end_coords = geocode_address(end)
        
        if not start_coords or not end_coords:
            return jsonify({'error': 'Could not find addresses. Please be more specific.'}), 400
        
        # Get route
        route = get_route_osrm(start_coords['lat'], start_coords['lon'], end_coords['lat'], end_coords['lon'])
        
        if not route:
            return jsonify({'error': 'Could not calculate route'}), 400
        
        # Predict risk using your model
        risk = predict_risk_with_model(
            route['distance_km'], route['duration_min'], 
            vehicle_type, time_of_day, weather
        )
        
        # Send email if provided
        email_sent = False
        if user_email and BREVO_API_KEY:
            email_sent = send_email_alert(
                risk['severity'], risk['confidence'], 
                risk['risk_factors'], user_email, risk['risk_level']
            )
        
        return jsonify({
            'success': True,
            'route': {
                'distance_km': route['distance_km'],
                'duration_min': route['duration_min'],
                'points': route['points']
            },
            'risk': risk,
            'email_sent': email_sent
        })
        
    except Exception as e:
        print(f"Error in calculate_route: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'features': features
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)