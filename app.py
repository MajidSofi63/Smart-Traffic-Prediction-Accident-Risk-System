from flask import Flask, render_template, request, jsonify, session
from flask import render_template_string
import pandas as pd
import numpy as np
import joblib
import os
import requests
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import json
from datetime import datetime

app = Flask(__name__)
app.secret_key = "traffic_prediction_secret_key_2024"

# Global variables for models
model = None
label_encoders = None
target_encoder = None
features = None

# Email configuration (for alerts)
EMAIL_CONFIG = {
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 587,
    'sender_email': 'your_email@gmail.com',  # Update with your email
    'sender_password': 'your_app_password'    # Update with app password
}

def load_models():
    """Load models with error handling"""
    global model, label_encoders, target_encoder, features
    
    try:
        print("Loading model...")
        model = joblib.load('model/traffic_model.pkl')
        print("✓ Model loaded")
        
        print("Loading encoders...")
        label_encoders = joblib.load('model/label_encoders.pkl')
        target_encoder = joblib.load('model/target_encoder.pkl')
        features = joblib.load('model/selected_features.pkl')
        print("✓ All models loaded successfully")
        print(f"✓ Expected features: {features}")
        print(f"✓ Target classes: {target_encoder.classes_.tolist()}")
        
        return True
    except Exception as e:
        print(f"✗ Error loading models: {e}")
        return False

def generate_explanation(features_dict, prediction, confidence):
    """Generate detailed explanation for why the prediction was made"""
    explanations = []
    recommendations = []
    risk_score = 0
    
    # Day of Week analysis
    if features_dict['Day_of_Week'] in ['Saturday', 'Sunday']:
        explanations.append("📅 Weekend travel - Accident rates are 35% higher on weekends due to increased traffic and recreational driving")
        risk_score += 25
        recommendations.append("Consider traveling on weekdays if possible")
    elif features_dict['Day_of_Week'] == 'Friday':
        explanations.append("📅 Friday evening - Peak accident time due to end-of-week rush")
        risk_score += 15
        recommendations.append("Avoid peak hours (5-7 PM) on Fridays")
    
    # Speed limit analysis
    speed = int(features_dict['Speed_limit'])
    if speed > 60:
        explanations.append(f"🏁 High speed limit ({speed} mph) - Higher speeds increase impact force and reduce reaction time")
        risk_score += 30
        recommendations.append(f"Reduce speed to 50 mph or below on this road")
    elif speed > 40:
        explanations.append(f"⚡ Moderate speed limit ({speed} mph) - Still requires caution")
        risk_score += 10
        recommendations.append("Maintain safe following distance")
    
    # Weather analysis
    weather = features_dict['Weather_Conditions']
    if 'Raining' in weather or 'Snowing' in weather:
        explanations.append(f"🌧️ Bad weather ({weather}) - Reduced visibility and traction, increased stopping distance by 50%")
        risk_score += 25
        recommendations.append("Use headlights, reduce speed, increase following distance")
    elif 'Fog' in weather:
        explanations.append(f"🌫️ Foggy conditions - Visibility reduced to less than 100 meters")
        risk_score += 30
        recommendations.append("Use fog lights, avoid sudden braking")
    
    # Light conditions analysis
    light = features_dict['Light_Conditions']
    if 'Darkness' in light:
        if 'no lighting' in light:
            explanations.append("🌙 Night travel on unlit roads - 3x higher fatality risk, limited visibility")
            risk_score += 35
            recommendations.append("Avoid unlit roads at night, use high beams when safe")
        else:
            explanations.append("🌃 Night travel - Reduced visibility, higher accident rates")
            risk_score += 20
            recommendations.append("Be extra cautious, watch for pedestrians and animals")
    
    # Vehicle analysis
    vehicles = int(features_dict['Number_of_Vehicles'])
    if vehicles > 3:
        explanations.append(f"🚗 Multi-vehicle collision risk ({vehicles} vehicles) - Complex accident scenario")
        risk_score += 25
        recommendations.append("Maintain extra distance, anticipate others' moves")
    elif vehicles > 2:
        explanations.append(f"🚙 Multiple vehicles involved ({vehicles}) - Increased collision probability")
        risk_score += 10
    
    # Casualties analysis
    casualties = int(features_dict['Number_of_Casualties'])
    if casualties > 2:
        explanations.append(f"⚠️ High casualty potential ({casualties} people) - Severe accident scenario")
        risk_score += 20
        recommendations.append("Ensure all passengers wear seatbelts/helmets")
    
    # Road surface analysis
    surface = features_dict['Road_Surface_Conditions']
    if surface in ['Wet or damp', 'Frost or ice', 'Snow']:
        explanations.append(f"⚠️ Slippery road surface ({surface}) - Increased stopping distance, risk of skidding")
        risk_score += 20
        recommendations.append("Reduce speed by 30% on slippery roads")
    
    # Area analysis
    area = features_dict['Urban_or_Rural_Area']
    if area == 'Rural':
        explanations.append("🏞️ Rural area - Higher speeds, less lighting, slower emergency response")
        risk_score += 15
        recommendations.append("Be aware of sharp curves, animals on road")
    
    # Junction analysis
    junction = features_dict['Junction_Control']
    if 'Auto traffic signal' in junction:
        explanations.append("🚦 Signal-controlled junction - Higher accident concentration at intersections")
        risk_score += 10
        recommendations.append("Be cautious at intersections, don't rush yellow lights")
    
    # Determine risk level
    if risk_score >= 60:
        risk_level = "HIGH"
        alert_needed = True
    elif risk_score >= 30:
        risk_level = "MEDIUM"
        alert_needed = False
    else:
        risk_level = "LOW"
        alert_needed = False
    
    # Add ML prediction context
    explanations.append(f"🤖 AI Model Confidence: {confidence}% based on {len(features)} factors")
    
    return {
        'explanations': explanations,
        'recommendations': recommendations[:5],  # Top 5 recommendations
        'risk_score': risk_score,
        'risk_level': risk_level,
        'alert_needed': alert_needed
    }

def send_email_alert(severity, confidence, explanations, user_email=None):
    """Send email alert for high-risk predictions"""
    try:
        subject = f"🚨 HIGH RISK ALERT: {severity} Accident Predicted"
        
        body = f"""
        <html>
        <body>
            <h2 style="color: red;">🚨 Traffic Risk Alert</h2>
            <p><strong>Prediction:</strong> {severity} Accident</p>
            <p><strong>Confidence:</strong> {confidence}%</p>
            
            <h3>⚠️ Risk Factors:</h3>
            <ul>
        """
        
        for exp in explanations[:5]:
            body += f"<li>{exp}</li>"
        
        body += """
            </ul>
            
            <h3>💡 Safety Recommendations:</h3>
            <ul>
        """
        
        # Add recommendations
        body += """
                <li>Consider postponing or changing your route</li>
                <li>Drive with extra caution</li>
                <li>Ensure all safety measures are followed</li>
            </ul>
            
            <p>Stay safe! 🚗</p>
        </body>
        </html>
        """
        
        # Send email (configure with your email settings)
        # msg = MIMEMultipart()
        # msg['Subject'] = subject
        # msg['From'] = EMAIL_CONFIG['sender_email']
        # msg['To'] = user_email or 'default_recipient@example.com'
        # msg.attach(MIMEText(body, 'html'))
        
        # server = smtplib.SMTP(EMAIL_CONFIG['smtp_server'], EMAIL_CONFIG['smtp_port'])
        # server.starttls()
        # server.login(EMAIL_CONFIG['sender_email'], EMAIL_CONFIG['sender_password'])
        # server.send_message(msg)
        # server.quit()
        
        print(f"📧 Alert email would be sent to {user_email or 'default'}")
        return True
    except Exception as e:
        print(f"Email error: {e}")
        return False

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/map')
def map_route():
    """Show interactive map for route planning"""
    return render_template('map_route.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('result.html', 
                             severity="Error", 
                             confidence=0,
                             error="Model not loaded")
    
    try:
        # Get all form data
        input_dict = {}
        for feature in features:
            value = request.form.get(feature)
            if value is None or value == '':
                return render_template('result.html', 
                                     severity="Error", 
                                     confidence=0,
                                     error=f"Missing value for {feature}")
            input_dict[feature] = value
        
        # Create dataframe
        input_df = pd.DataFrame([input_dict])
        
        # Convert numeric columns
        numeric_cols = ['Speed_limit', 'Number_of_Vehicles', 'Number_of_Casualties']
        for col in numeric_cols:
            if col in input_df.columns:
                input_df[col] = pd.to_numeric(input_df[col])
        
        # Encode categorical variables
        for column in input_df.columns:
            if column in label_encoders:
                le = label_encoders[column]
                try:
                    input_df[column] = le.transform(input_df[column].astype(str))
                except:
                    input_df[column] = 0
        
        # Make prediction
        prediction = model.predict(input_df)
        predicted_severity = target_encoder.inverse_transform(prediction)[0]
        
        # Get confidence
        probabilities = model.predict_proba(input_df)[0]
        confidence = max(probabilities) * 100
        
        # Generate explanation
        explanation_data = generate_explanation(input_dict, predicted_severity, confidence)
        
        # Send alert if high risk
        alert_sent = False
        if explanation_data['alert_needed'] and predicted_severity in ['Serious', 'Fatal']:
            user_email = request.form.get('user_email')  # Optional email field
            alert_sent = send_email_alert(predicted_severity, confidence, 
                                         explanation_data['explanations'], user_email)
        
        return render_template('result.html', 
                             severity=predicted_severity,
                             confidence=round(confidence, 2),
                             explanations=explanation_data['explanations'],
                             recommendations=explanation_data['recommendations'],
                             risk_level=explanation_data['risk_level'],
                             risk_score=explanation_data['risk_score'],
                             alert_sent=alert_sent,
                             error=None)
    
    except Exception as e:
        return render_template('result.html', 
                             severity="Error", 
                             confidence=0,
                             error=str(e))

@app.route('/predict_route_risk', methods=['POST'])
def predict_route_risk():
    """Predict risk for a specific route segment"""
    try:
        data = request.json
        
        input_data = {
            'Day_of_Week': data.get('day_of_week', 'Monday'),
            'Junction_Control': data.get('junction_control', 'Not at junction or within 20m'),
            'Light_Conditions': data.get('light_conditions', 'Daylight'),
            'Road_Surface_Conditions': data.get('road_surface', 'Dry'),
            'Road_Type': data.get('road_type', 'Single carriageway'),
            'Speed_limit': data.get('speed_limit', 50),
            'Urban_or_Rural_Area': data.get('area_type', 'Urban'),
            'Weather_Conditions': data.get('weather', 'Fine no high winds'),
            'Number_of_Vehicles': data.get('vehicles', 2),
            'Number_of_Casualties': data.get('casualties', 1)
        }
        
        input_df = pd.DataFrame([input_data])
        
        for column in input_df.columns:
            if column in label_encoders:
                le = label_encoders[column]
                try:
                    input_df[column] = le.transform(input_df[column].astype(str))
                except:
                    input_df[column] = 0
        
        prediction = model.predict(input_df)
        predicted_severity = target_encoder.inverse_transform(prediction)[0]
        
        probabilities = model.predict_proba(input_df)[0]
        confidence = max(probabilities) * 100
        
        # Generate explanation for route
        explanation_data = generate_explanation(input_data, predicted_severity, confidence)
        
        return jsonify({
            'severity': predicted_severity,
            'confidence': round(confidence, 2),
            'explanations': explanation_data['explanations'][:3],
            'recommendations': explanation_data['recommendations'][:3],
            'risk_level': explanation_data['risk_level'],
            'alert_needed': explanation_data['alert_needed']
        })
        
    except Exception as e:
        return jsonify({'severity': 'Unknown', 'confidence': 0, 'error': str(e)})

@app.route('/geocode', methods=['GET', 'POST'])
def geocode():
    """Geocode an address to coordinates (supports manual input)"""
    address = request.args.get('address') or request.json.get('address') if request.json else None
    
    if not address:
        return jsonify({'error': 'No address provided'})
    
    url = f"https://nominatim.openstreetmap.org/search?format=json&q={address}&limit=1"
    try:
        response = requests.get(url, headers={'User-Agent': 'TrafficRiskApp/1.0'})
        data = response.json()
        
        if data:
            return jsonify({
                'lat': float(data[0]['lat']),
                'lon': float(data[0]['lon']),
                'display_name': data[0]['display_name']
            })
    except Exception as e:
        return jsonify({'error': str(e)})
    
    return jsonify({'error': 'Location not found'})

@app.route('/health')
def health_check():
    return jsonify({"status": "healthy", "models_loaded": model is not None})

if __name__ == '__main__':
    load_models()
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)