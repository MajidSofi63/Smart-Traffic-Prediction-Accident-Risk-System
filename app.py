from flask import Flask, render_template, request, jsonify
from flask import render_template_string
import pandas as pd
import numpy as np
import joblib
import os
import sys
import requests

app = Flask(__name__)

# Global variables for models
model = None
label_encoders = None
target_encoder = None
features = None

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

# Load models when the app starts
with app.app_context():
    load_models()

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
        
        # Create dataframe
        input_df = pd.DataFrame([input_dict])
        
        # Convert numeric columns
        numeric_cols = ['Speed_limit', 'Number_of_Vehicles', 'Number_of_Casualties']
        for col in numeric_cols:
            if col in input_df.columns:
                try:
                    input_df[col] = pd.to_numeric(input_df[col])
                except:
                    return render_template('result.html', 
                                         severity="Error", 
                                         confidence=0,
                                         error=f"Invalid numeric value for {col}")
        
        # Encode categorical variables
        for column in input_df.columns:
            if column in label_encoders:
                le = label_encoders[column]
                try:
                    input_df[column] = le.transform(input_df[column].astype(str))
                except ValueError as e:
                    return render_template('result.html', 
                                         severity="Error", 
                                         confidence=0,
                                         error=f"Invalid value for {column}")
        
        # Make prediction
        prediction = model.predict(input_df)
        predicted_severity = target_encoder.inverse_transform(prediction)[0]
        
        # Get prediction probabilities
        probabilities = model.predict_proba(input_df)[0]
        confidence = max(probabilities) * 100
        
        return render_template('result.html', 
                             severity=predicted_severity,
                             confidence=round(confidence, 2),
                             error=None)
    
    except Exception as e:
        print(f"Error details: {str(e)}")
        return render_template('result.html', 
                             severity="Error", 
                             confidence=0,
                             error=f"Prediction error: {str(e)}")

@app.route('/predict_route_risk', methods=['POST'])
def predict_route_risk():
    """Predict risk for a specific route segment"""
    try:
        data = request.json
        
        # Prepare features for ML model
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
        
        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])
        
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
        
        return jsonify({
            'severity': predicted_severity,
            'confidence': round(confidence, 2)
        })
        
    except Exception as e:
        return jsonify({'severity': 'Unknown', 'confidence': 0, 'error': str(e)})

@app.route('/geocode', methods=['GET'])
def geocode():
    """Geocode an address to coordinates"""
    address = request.args.get('address', '')
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

@app.route('/debug')
def debug():
    """Debug endpoint to see all valid values for each field"""
    if model is None:
        return jsonify({'error': 'Model not loaded'})
    
    debug_info = {}
    for col, encoder in label_encoders.items():
        debug_info[col] = {
            'type': 'categorical',
            'valid_values': encoder.classes_.tolist()
        }
    
    return jsonify(debug_info)

@app.route('/health')
def health_check():
    """Health check endpoint for Render"""
    return jsonify({
        "status": "healthy",
        "models_loaded": model is not None,
        "features": features
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug_mode)@app.route('/test_map')
def test_map():
    return "<h1>Map route working!</h1><p>If you see this, the route is accessible.</p><a href='/map'>Go to Map</a>"