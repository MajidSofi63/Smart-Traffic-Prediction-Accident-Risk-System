from flask import Flask, render_template, request, jsonify
from flask import render_template_string
import pandas as pd
import numpy as np
import joblib
import os
import sys

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
    if not load_models():
        print("WARNING: Models failed to load. The app may not work correctly.")

@app.route('/')
def home():
    return render_template('index.html')

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
                    valid_values = le.classes_.tolist()
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
    
    # Add numeric columns info
    numeric_cols = ['Speed_limit', 'Number_of_Vehicles', 'Number_of_Casualties']
    for col in numeric_cols:
        if col in features:
            debug_info[col] = {
                'type': 'numeric',
                'valid_values': 'any integer value'
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

@app.route('/valid_values')
def valid_values():
    """Show all valid values for each field in HTML format"""
    if model is None:
        return "<html><body><h1>Model not loaded. Please check server logs.</h1></body></html>"
    
    valid_values_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Valid Values for Accident Prediction System</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
            .container { background-color: white; padding: 30px; border-radius: 15px; }
            h1 { color: #333; text-align: center; }
            h2 { color: #667eea; border-bottom: 2px solid #667eea; padding-bottom: 10px; margin-top: 30px; }
            ul { list-style-type: none; padding: 0; }
            li { background-color: #f5f5f5; margin: 8px 0; padding: 10px; border-radius: 5px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Valid Values for Accident Prediction System</h1>
    """
    
    for col, encoder in label_encoders.items():
        valid_values_html += f"<h2>{col}</h2><ul>"
        for val in encoder.classes_:
            valid_values_html += f"<li>{val}</li>"
        valid_values_html += "</ul>"
    
    valid_values_html += "</div></body></html>"
    return valid_values_html

@app.route('/dynamic_form')
def dynamic_form():
    """Dynamic form that loads valid values directly from the model"""
    if model is None:
        return "<html><body><h1>Model not loaded. Please check server logs.</h1></body></html>"
    
    # Get valid values for each field
    field_values = {}
    for col, encoder in label_encoders.items():
        field_values[col] = encoder.classes_.tolist()
    
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Accident Risk Prediction System - Dynamic Form</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
            .container { background-color: white; padding: 30px; border-radius: 15px; }
            h1 { color: #333; text-align: center; }
            .form-group { margin-bottom: 20px; }
            label { display: block; margin-bottom: 8px; font-weight: bold; }
            select, input { width: 100%; padding: 10px; border: 2px solid #ddd; border-radius: 8px; }
            button { width: 100%; padding: 12px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; border-radius: 8px; cursor: pointer; font-size: 16px; }
            .row { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
            h3 { color: #667eea; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Smart Traffic Prediction & Accident Risk System</h1>
            <form action="/predict" method="post">
                <div class="row">
                    <div class="form-group">
                        <label>Day of Week:</label>
                        <select name="Day_of_Week" required>
                            {% for val in field_values['Day_of_Week'] %}
                            <option value="{{ val }}">{{ val }}</option>
                            {% endfor %}
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Urban or Rural Area:</label>
                        <select name="Urban_or_Rural_Area" required>
                            {% for val in field_values['Urban_or_Rural_Area'] %}
                            <option value="{{ val }}">{{ val }}</option>
                            {% endfor %}
                        </select>
                    </div>
                </div>
                <div class="row">
                    <div class="form-group">
                        <label>Junction Control:</label>
                        <select name="Junction_Control" required>
                            {% for val in field_values['Junction_Control'] %}
                            <option value="{{ val }}">{{ val }}</option>
                            {% endfor %}
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Road Type:</label>
                        <select name="Road_Type" required>
                            {% for val in field_values['Road_Type'] %}
                            <option value="{{ val }}">{{ val }}</option>
                            {% endfor %}
                        </select>
                    </div>
                </div>
                <div class="row">
                    <div class="form-group">
                        <label>Road Surface Conditions:</label>
                        <select name="Road_Surface_Conditions" required>
                            {% for val in field_values['Road_Surface_Conditions'] %}
                            <option value="{{ val }}">{{ val }}</option>
                            {% endfor %}
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Speed Limit (mph):</label>
                        <input type="number" name="Speed_limit" value="30" required>
                    </div>
                </div>
                <div class="row">
                    <div class="form-group">
                        <label>Light Conditions:</label>
                        <select name="Light_Conditions" required>
                            {% for val in field_values['Light_Conditions'] %}
                            <option value="{{ val }}">{{ val }}</option>
                            {% endfor %}
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Weather Conditions:</label>
                        <select name="Weather_Conditions" required>
                            {% for val in field_values['Weather_Conditions'] %}
                            <option value="{{ val }}">{{ val }}</option>
                            {% endfor %}
                        </select>
                    </div>
                </div>
                <div class="row">
                    <div class="form-group">
                        <label>Number of Vehicles:</label>
                        <input type="number" name="Number_of_Vehicles" min="1" value="2" required>
                    </div>
                    <div class="form-group">
                        <label>Number of Casualties:</label>
                        <input type="number" name="Number_of_Casualties" min="0" value="1" required>
                    </div>
                </div>
                <button type="submit">Predict Accident Severity</button>
            </form>
        </div>
    </body>
    </html>
    ''', field_values=field_values)

if __name__ == '__main__':
    # Get port from environment variable or default to 10000
    port = int(os.environ.get('PORT', 10000))
    # Disable debug mode in production
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug_mode)

@app.route('/map')
def map_route():
    """Show interactive map for route planning"""
    return render_template('map_route.html')

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
    import requests
    address = request.args.get('address', '')
    if not address:
        return jsonify({'error': 'No address provided'})
    
    url = f"https://nominatim.openstreetmap.org/search?format=json&q={address}&limit=1"
    response = requests.get(url, headers={'User-Agent': 'TrafficRiskApp/1.0'})
    data = response.json()
    
    if data:
        return jsonify({
            'lat': float(data[0]['lat']),
            'lon': float(data[0]['lon']),
            'display_name': data[0]['display_name']
        })
    return jsonify({'error': 'Location not found'})    