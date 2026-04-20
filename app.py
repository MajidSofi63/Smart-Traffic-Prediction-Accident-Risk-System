from flask import Flask, render_template, request, jsonify, session
from flask import render_template_string
import pandas as pd
import numpy as np
import joblib
import os
import sys
import requests
import sib_api_v3_sdk
from sib_api_v3_sdk.rest import ApiException
from datetime import datetime

app = Flask(__name__)
app.secret_key = "traffic_prediction_secret_key_2024"

# Global variables for models
model = None
label_encoders = None
target_encoder = None
features = None

# Brevo Configuration (300 free emails/day)
BREVO_CONFIG = {
    'api_key': os.environ.get('BREVO_API_KEY'),
    'sender_email': os.environ.get('BREVO_SENDER_EMAIL', 'majidsofi63@gmail.com'),
    'sender_name': 'Smart Traffic Prediction System'
}

def load_models():
    """Load models with error handling"""
    global model, label_encoders, target_encoder, features
    
    try:
        print(f"Current working directory: {os.getcwd()}")
        print(f"Model directory exists: {os.path.exists('model')}")
        
        if os.path.exists('model'):
            print(f"Files in model directory: {os.listdir('model')}")
        
        print("Loading model...")
        model = joblib.load('model/traffic_model.pkl')
        print("✓ Model loaded - type:", type(model))
        
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
        import traceback
        traceback.print_exc()
        return False

def generate_explanation(features_dict, prediction, confidence):
    """Generate detailed explanation for why the prediction was made"""
    explanations = []
    recommendations = []
    risk_score = 0
    
    # Day of Week analysis
    if features_dict['Day_of_Week'] in ['Saturday', 'Sunday']:
        explanations.append("📅 Weekend travel - Accident rates are 35% higher on weekends")
        risk_score += 25
        recommendations.append("Consider traveling on weekdays if possible")
    elif features_dict['Day_of_Week'] == 'Friday':
        explanations.append("📅 Friday evening - Peak accident time due to end-of-week rush")
        risk_score += 15
        recommendations.append("Avoid peak hours (5-7 PM) on Fridays")
    
    # Speed limit analysis
    speed = int(features_dict['Speed_limit'])
    if speed > 60:
        explanations.append(f"🏁 High speed limit ({speed} mph) - Higher impact force")
        risk_score += 30
        recommendations.append(f"Reduce speed to 50 mph or below")
    elif speed > 40:
        explanations.append(f"⚡ Moderate speed limit ({speed} mph)")
        risk_score += 10
        recommendations.append("Maintain safe following distance")
    
    # Weather analysis
    weather = features_dict['Weather_Conditions']
    if 'Raining' in weather or 'Snowing' in weather:
        explanations.append(f"🌧️ Bad weather ({weather}) - Reduced visibility and traction")
        risk_score += 25
        recommendations.append("Use headlights, reduce speed")
    elif 'Fog' in weather:
        explanations.append(f"🌫️ Foggy conditions - Visibility reduced")
        risk_score += 30
        recommendations.append("Use fog lights, avoid sudden braking")
    
    # Light conditions analysis
    light = features_dict['Light_Conditions']
    if 'Darkness' in light:
        if 'no lighting' in light:
            explanations.append("🌙 Night on unlit roads - 3x higher fatality risk")
            risk_score += 35
            recommendations.append("Avoid unlit roads at night")
        else:
            explanations.append("🌃 Night travel - Reduced visibility")
            risk_score += 20
            recommendations.append("Be extra cautious")
    
    # Vehicle analysis
    vehicles = int(features_dict['Number_of_Vehicles'])
    if vehicles > 3:
        explanations.append(f"🚗 Multi-vehicle collision ({vehicles} vehicles)")
        risk_score += 25
        recommendations.append("Maintain extra distance")
    elif vehicles > 2:
        explanations.append(f"🚙 Multiple vehicles ({vehicles})")
        risk_score += 10
    
    # Casualties analysis
    casualties = int(features_dict['Number_of_Casualties'])
    if casualties > 2:
        explanations.append(f"⚠️ High casualty potential ({casualties} people)")
        risk_score += 20
        recommendations.append("Ensure all wear seatbelts/helmets")
    
    # Road surface analysis
    surface = features_dict['Road_Surface_Conditions']
    if surface in ['Wet or damp', 'Frost or ice', 'Snow']:
        explanations.append(f"⚠️ Slippery road ({surface})")
        risk_score += 20
        recommendations.append("Reduce speed by 30% on slippery roads")
    
    # Area analysis
    area = features_dict['Urban_or_Rural_Area']
    if area == 'Rural':
        explanations.append("🏞️ Rural area - Higher speeds, slower emergency response")
        risk_score += 15
        recommendations.append("Watch for curves, animals")
    
    # Junction analysis
    junction = features_dict['Junction_Control']
    if 'Auto traffic signal' in junction:
        explanations.append("🚦 Signal-controlled junction - Higher accident concentration")
        risk_score += 10
        recommendations.append("Be cautious at intersections")
    
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
    
    explanations.append(f"🤖 AI Confidence: {confidence}%")
    
    return {
        'explanations': explanations,
        'recommendations': recommendations[:5],
        'risk_score': risk_score,
        'risk_level': risk_level,
        'alert_needed': alert_needed
    }

def send_email_alert(severity, confidence, explanations, user_email=None, risk_level="MEDIUM"):
    """Send email alert for EVERY trip"""
    if not user_email:
        print("📧 No email provided")
        return False
    
    if not BREVO_CONFIG['api_key'] or BREVO_CONFIG['api_key'] == 'your_brevo_api_key_here':
        print("⚠️ Brevo not configured")
        return False
    
    try:
        configuration = sib_api_v3_sdk.Configuration()
        configuration.api_key['api-key'] = BREVO_CONFIG['api_key']
        api_instance = sib_api_v3_sdk.TransactionalEmailsApi(sib_api_v3_sdk.ApiClient(configuration))
        
        if risk_level == "HIGH":
            subject = f"🚨 HIGH RISK ALERT: {severity} Accident Predicted"
        elif risk_level == "MEDIUM":
            subject = f"⚠️ MEDIUM RISK: {severity} Accident Possible"
        else:
            subject = f"✅ TRIP SUMMARY: {severity} Risk Assessment"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head><meta charset="UTF-8"></head>
        <body style="font-family: Arial, sans-serif;">
            <h2>🚗 Smart Traffic Prediction System</h2>
            <h3>Prediction: {severity} Accident ({confidence}% confidence)</h3>
            <p><strong>Risk Level:</strong> {risk_level}</p>
            <h3>🔍 Risk Factors:</h3>
            <ul>
        """
        for exp in explanations[:5]:
            html_content += f"<li>{exp}</li>"
        
        html_content += """
            </ul>
            <h3>💡 Recommendations:</h3>
            <ul>
                <li>Drive cautiously</li>
                <li>Follow traffic rules</li>
                <li>Stay alert</li>
            </ul>
            <p>Stay safe! 🚗</p>
        </body>
        </html>
        """
        
        sender = {"name": BREVO_CONFIG['sender_name'], "email": BREVO_CONFIG['sender_email']}
        to = [{"email": user_email}]
        
        send_smtp_email = sib_api_v3_sdk.SendSmtpEmail(
            to=to, sender=sender, subject=subject, html_content=html_content
        )
        
        api_instance.send_transac_email(send_smtp_email)
        print(f"✅ Email sent to {user_email} - Risk: {risk_level}")
        return True
    except Exception as e:
        print(f"❌ Email error: {e}")
        return False

# Load models
with app.app_context():
    load_models()

# ========== ROUTES ==========

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/map')
def map_route():
    return render_template('map_route.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Predict accident severity - SINGLE VERSION"""
    if model is None:
        return render_template('result.html', severity="Error", confidence=0, error="Model not loaded")
    
    try:
        input_dict = {}
        for feature in features:
            value = request.form.get(feature)
            if not value:
                return render_template('result.html', severity="Error", confidence=0, error=f"Missing {feature}")
            input_dict[feature] = value
        
        user_email = request.form.get('user_email', '')
        input_df = pd.DataFrame([input_dict])
        
        numeric_cols = ['Speed_limit', 'Number_of_Vehicles', 'Number_of_Casualties']
        for col in numeric_cols:
            if col in input_df.columns:
                input_df[col] = pd.to_numeric(input_df[col])
        
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
        
        explanation_data = generate_explanation(input_dict, predicted_severity, confidence)
        
        # Send email for EVERY trip (if email provided)
        alert_sent = False
        if user_email:
            alert_sent = send_email_alert(predicted_severity, confidence, 
                                         explanation_data['explanations'], 
                                         user_email, explanation_data['risk_level'])
        
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
        return render_template('result.html', severity="Error", confidence=0, error=str(e))

@app.route('/predict_route_risk', methods=['POST'])
def predict_route_risk():
    """Predict risk for route"""
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
        
        user_email = data.get('user_email', '')
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
        
        explanation_data = generate_explanation(input_data, predicted_severity, confidence)
        
        alert_sent = False
        if user_email:
            alert_sent = send_email_alert(predicted_severity, confidence, 
                                         explanation_data['explanations'], 
                                         user_email, explanation_data['risk_level'])
        
        return jsonify({
            'severity': predicted_severity,
            'confidence': round(confidence, 2),
            'explanations': explanation_data['explanations'][:3],
            'recommendations': explanation_data['recommendations'][:3],
            'risk_level': explanation_data['risk_level'],
            'risk_score': explanation_data['risk_score'],
            'alert_sent': alert_sent
        })
    except Exception as e:
        return jsonify({'severity': 'Unknown', 'confidence': 0, 'error': str(e)})

@app.route('/geocode', methods=['GET', 'POST'])
def geocode():
    """Geocode address to coordinates"""
    try:
        if request.method == 'POST':
            data = request.get_json()
            address = data.get('address') if data else None
        else:
            address = request.args.get('address')
        
        if not address:
            return jsonify({'error': 'No address provided'}), 400
        
        url = f"https://nominatim.openstreetmap.org/search?format=json&q={address}&limit=1"
        response = requests.get(url, headers={'User-Agent': 'SmartTrafficPrediction/1.0'}, timeout=10)
        
        if response.status_code != 200:
            return jsonify({'error': 'Geocoding service error'}), 500
        
        data = response.json()
        if data:
            return jsonify({'lat': float(data[0]['lat']), 'lon': float(data[0]['lon']), 'display_name': data[0]['display_name']})
        return jsonify({'error': f'Address not found: {address}'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    return jsonify({"status": "healthy", "models_loaded": model is not None, "features": features})

@app.route('/debug_model')
def debug_model():
    import os
    return jsonify({
        'cwd': os.getcwd(),
        'model_dir_exists': os.path.exists('model'),
        'model_files': os.listdir('model') if os.path.exists('model') else [],
        'model_loaded': model is not None,
        'features': features,
        'python_version': sys.version
    })

@app.route('/debug')
def debug():
    if model is None:
        return jsonify({'error': 'Model not loaded'})
    debug_info = {"model_loaded": True, "features": features, "target_classes": target_encoder.classes_.tolist()}
    for col, encoder in label_encoders.items():
        debug_info[col] = encoder.classes_.tolist()
    return jsonify(debug_info)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)