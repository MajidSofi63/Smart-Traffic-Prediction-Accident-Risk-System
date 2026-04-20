from flask import Flask, render_template, request, jsonify, session
from flask import render_template_string
import pandas as pd
import numpy as np
import joblib
import os
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
# Get your free API key from: https://www.brevo.com
BREVO_CONFIG = {
    'api_key': os.environ.get('BREVO_API_KEY'),  # Reads from environment
    'sender_email': os.environ.get('BREVO_SENDER_EMAIL', 'majidsofi63@gmail.com'),
    'sender_name': 'Smart Traffic Prediction System'
}

def load_models():
    """Load models with error handling"""
    global model, label_encoders, target_encoder, features
    
    try:
        import os
        print(f"Current working directory: {os.getcwd()}")
        print(f"Files in model directory: {os.listdir('model') if os.path.exists('model') else 'model folder not found'}")
        
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
        explanations.append("📅 Weekend travel - Accident rates are 35% higher on weekends due to increased traffic and recreational driving")
        risk_score += 25
        recommendations.append("Consider traveling on weekdays if possible")
    elif features_dict['Day_of_Week'] == 'Friday':
        explanations.append("📅 Friday evening - Peak accident time due to end-of-week rush")
        risk_score += 15
        recommendations.append("Avoid peak hours (5-7 PM) on Fridays")
    else:
        explanations.append(f"📅 {features_dict['Day_of_Week']} - Weekday travel has lower accident rates")
        recommendations.append("Maintain regular safety practices")
    
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
    else:
        explanations.append(f"🐌 Low speed limit ({speed} mph) - Safer speed zone")
        recommendations.append("Continue maintaining safe speed")
    
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
    else:
        explanations.append(f"☀️ Good weather ({weather}) - Favorable driving conditions")
        recommendations.append("Enjoy safe driving conditions")
    
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
    else:
        explanations.append("☀️ Daylight travel - Optimal visibility conditions")
        recommendations.append("Good visibility, maintain standard precautions")
    
    # Vehicle analysis
    vehicles = int(features_dict['Number_of_Vehicles'])
    if vehicles > 3:
        explanations.append(f"🚗 Multi-vehicle collision risk ({vehicles} vehicles) - Complex accident scenario")
        risk_score += 25
        recommendations.append("Maintain extra distance, anticipate others' moves")
    elif vehicles > 2:
        explanations.append(f"🚙 Multiple vehicles involved ({vehicles}) - Increased collision probability")
        risk_score += 10
        recommendations.append("Stay alert, watch surrounding traffic")
    else:
        explanations.append(f"🚘 Single vehicle involvement ({vehicles}) - Lower collision complexity")
        recommendations.append("Focus on road and driving conditions")
    
    # Casualties analysis
    casualties = int(features_dict['Number_of_Casualties'])
    if casualties > 2:
        explanations.append(f"⚠️ High casualty potential ({casualties} people) - Severe accident scenario")
        risk_score += 20
        recommendations.append("Ensure all passengers wear seatbelts/helmets")
    elif casualties > 0:
        explanations.append(f"👥 Casualties possible ({casualties} people) - Moderate severity potential")
        recommendations.append("Drive defensively, protect all occupants")
    
    # Road surface analysis
    surface = features_dict['Road_Surface_Conditions']
    if surface in ['Wet or damp', 'Frost or ice', 'Snow']:
        explanations.append(f"⚠️ Slippery road surface ({surface}) - Increased stopping distance, risk of skidding")
        risk_score += 20
        recommendations.append("Reduce speed by 30% on slippery roads")
    else:
        explanations.append(f"✅ Good road surface ({surface}) - Proper traction available")
        recommendations.append("Normal driving conditions")
    
    # Area analysis
    area = features_dict['Urban_or_Rural_Area']
    if area == 'Rural':
        explanations.append("🏞️ Rural area - Higher speeds, less lighting, slower emergency response")
        risk_score += 15
        recommendations.append("Be aware of sharp curves, animals on road")
    else:
        explanations.append("🏙️ Urban area - Lower speeds, better lighting, faster emergency response")
        recommendations.append("Watch for pedestrians, cyclists, and intersections")
    
    # Junction analysis
    junction = features_dict['Junction_Control']
    if 'Auto traffic signal' in junction:
        explanations.append("🚦 Signal-controlled junction - Higher accident concentration at intersections")
        risk_score += 10
        recommendations.append("Be cautious at intersections, don't rush yellow lights")
    elif 'Give way' in junction:
        explanations.append("🛑 Give way junction - Right-of-way confusion possible")
        risk_score += 5
        recommendations.append("Always yield properly at junctions")
    
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
    explanations.append(f"🤖 AI Model Confidence: {confidence}% based on {len(features_dict)} factors")
    
    return {
        'explanations': explanations,
        'recommendations': recommendations[:5],
        'risk_score': risk_score,
        'risk_level': risk_level,
        'alert_needed': alert_needed
    }

def send_email_alert_brevo(severity, confidence, explanations, user_email=None):
    """
    Send email alert using Brevo API (300 free emails/day)
    This is the recommended email service for production use
    """
    # Skip if no email provided
    if not user_email:
        print("📧 No email provided, skipping email alert")
        return False
    
    # Check if Brevo is configured
    if BREVO_CONFIG['api_key'] == 'your_brevo_api_key_here':
        print("⚠️ Brevo not configured. Get a free API key from https://www.brevo.com")
        print("   Then set BREVO_API_KEY environment variable")
        return False
    
    try:
        # Configure API key
        configuration = sib_api_v3_sdk.Configuration()
        configuration.api_key['api-key'] = BREVO_CONFIG['api_key']
        
        # Create API instance
        api_instance = sib_api_v3_sdk.TransactionalEmailsApi(
            sib_api_v3_sdk.ApiClient(configuration)
        )
        
        # Build email content
        subject = f"🚨 HIGH RISK ALERT: {severity} Accident Predicted"
        
        # Determine risk color
        if severity == "Fatal":
            risk_color = "#f44336"
        elif severity == "Serious":
            risk_color = "#ff9800"
        else:
            risk_color = "#4CAF50"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; text-align: center; border-radius: 10px 10px 0 0; }}
                .risk-box {{ background: #ffebee; border-left: 4px solid {risk_color}; padding: 15px; margin: 20px 0; border-radius: 5px; }}
                .risk-level {{ font-size: 24px; font-weight: bold; color: {risk_color}; }}
                .explanations {{ background: #f5f5f5; padding: 15px; margin: 20px 0; border-radius: 5px; }}
                .recommendations {{ background: #e8f5e9; padding: 15px; margin: 20px 0; border-radius: 5px; }}
                .footer {{ font-size: 12px; color: #666; text-align: center; margin-top: 30px; padding-top: 20px; border-top: 1px solid #eee; }}
                ul {{ margin: 10px 0; padding-left: 20px; }}
                li {{ margin: 8px 0; }}
                .badge {{ display: inline-block; background: {risk_color}; color: white; padding: 5px 10px; border-radius: 20px; font-size: 12px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h2>🚗 Smart Traffic Prediction System</h2>
                    <p>Accident Risk Alert</p>
                </div>
                
                <div class="risk-box">
                    <div class="risk-level">⚠️ {severity} Accident Risk</div>
                    <p><strong>Confidence:</strong> {confidence}%</p>
                    <p><span class="badge">High Priority Alert</span></p>
                </div>
                
                <div class="explanations">
                    <h3>🔍 Risk Factors Identified:</h3>
                    <ul>
        """
        
        for exp in explanations[:6]:
            html_content += f"<li>{exp}</li>"
        
        html_content += """
                    </ul>
                </div>
                
                <div class="recommendations">
                    <h3>💡 Safety Recommendations:</h3>
                    <ul>
                        <li>🚨 Consider postponing or changing your route</li>
                        <li>⚠️ Drive with extra caution and reduce speed</li>
                        <li>🛡️ Ensure all safety measures are followed</li>
                        <li>🐌 Increase following distance to 4+ seconds</li>
                        <li>📞 Inform someone about your travel plans</li>
                    </ul>
                </div>
                
                <div class="footer">
                    <p>This is an automated alert from Smart Traffic Prediction System.</p>
                    <p>Powered by Brevo - 300 free emails/day</p>
                    <p>Stay safe! 🚗</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Create email object
        sender = {
            "name": BREVO_CONFIG['sender_name'],
            "email": BREVO_CONFIG['sender_email']
        }
        
        to = [{"email": user_email}]
        
        send_smtp_email = sib_api_v3_sdk.SendSmtpEmail(
            to=to,
            sender=sender,
            subject=subject,
            html_content=html_content
        )
        
        # Send email
        api_response = api_instance.send_transac_email(send_smtp_email)
        print(f"✅ Brevo alert sent successfully to {user_email}")
        print(f"   Message ID: {api_response.message_id}")
        return True
        
    except ApiException as e:
        print(f"❌ Brevo API error: {e}")
        print(f"   Status code: {e.status}")
        print(f"   Reason: {e.reason}")
        return False
    except Exception as e:
        print(f"❌ Email error: {e}")
        return False

# Use Brevo as the primary email service
send_email_alert = send_email_alert_brevo

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
                             error="Model not loaded. Please check server logs.")
    
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
        
        # Get email for alerts (optional)
        user_email = request.form.get('user_email', '')
        
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
        print(f"Prediction error: {str(e)}")
        return render_template('result.html', 
                             severity="Error", 
                             confidence=0,
                             error=f"Prediction error: {str(e)}")

@app.route('/predict_route_risk', methods=['POST'])
def predict_route_risk():
    """Predict risk for a specific route segment"""
    try:
        data = request.json
        
        # Get day of week based on time if not provided
        day_of_week = data.get('day_of_week', 'Monday')
        light_conditions = data.get('light_conditions', 'Daylight')
        
        input_data = {
            'Day_of_Week': day_of_week,
            'Junction_Control': data.get('junction_control', 'Not at junction or within 20m'),
            'Light_Conditions': light_conditions,
            'Road_Surface_Conditions': data.get('road_surface', 'Dry'),
            'Road_Type': data.get('road_type', 'Single carriageway'),
            'Speed_limit': data.get('speed_limit', 50),
            'Urban_or_Rural_Area': data.get('area_type', 'Urban'),
            'Weather_Conditions': data.get('weather', 'Fine no high winds'),
            'Number_of_Vehicles': data.get('vehicles', 2),
            'Number_of_Casualties': data.get('casualties', 1)
        }
        
        # Get user email for alerts
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
        
        # Send alert if high risk
        alert_sent = False
        if explanation_data['alert_needed'] and predicted_severity in ['Serious', 'Fatal']:
            alert_sent = send_email_alert(predicted_severity, confidence, 
                                         explanation_data['explanations'], user_email)
        
        return jsonify({
            'severity': predicted_severity,
            'confidence': round(confidence, 2),
            'explanations': explanation_data['explanations'][:3],
            'recommendations': explanation_data['recommendations'][:3],
            'risk_level': explanation_data['risk_level'],
            'risk_score': explanation_data['risk_score'],
            'alert_needed': explanation_data['alert_needed'],
            'alert_sent': alert_sent
        })
        
    except Exception as e:
        print(f"Route risk error: {str(e)}")
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
    """Health check endpoint for Render"""
    return jsonify({
        "status": "healthy", 
        "models_loaded": model is not None,
        "email_service": "Brevo (300 free emails/day)",
        "features": features
    })

@app.route('/debug')
def debug():
    """Debug endpoint to see all valid values for each field"""
    if model is None:
        return jsonify({'error': 'Model not loaded'})
    
    debug_info = {
        "model_loaded": True,
        "features": features,
        "target_classes": target_encoder.classes_.tolist(),
        "email_config": {
            "service": "Brevo",
            "configured": BREVO_CONFIG['api_key'] != 'your_brevo_api_key_here',
            "free_emails_per_day": 300
        }
    }
    
    for col, encoder in label_encoders.items():
        debug_info[col] = {
            'type': 'categorical',
            'valid_values': encoder.classes_.tolist()
        }
    
    return jsonify(debug_info)

if __name__ == '__main__':
    # Load models before starting
    load_models()
    
    # Get port from environment variable
    port = int(os.environ.get('PORT', 10000))
    
    # Run the app (debug=False for production)
    app.run(host='0.0.0.0', port=port, debug=False)