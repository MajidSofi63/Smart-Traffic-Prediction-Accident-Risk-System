from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)

# Load the trained model and encoders
try:
    model = joblib.load('model/traffic_model.pkl')
    label_encoders = joblib.load('model/label_encoders.pkl')
    target_encoder = joblib.load('model/target_encoder.pkl')
    features = joblib.load('model/selected_features.pkl')
    print("✓ Model loaded successfully!")
    print(f"✓ Expected features: {features}")
    print(f"✓ Target classes: {target_encoder.classes_.tolist()}")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('result.html', 
                             severity="Error", 
                             confidence=0,
                             error="Model not loaded. Please train the model first by running: python model/train.py")
    
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
                                         error=f"Invalid value '{input_df[column].iloc[0]}' for {column}. Valid values: {', '.join(valid_values[:10])}")
        
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

@app.route('/valid_values')
def valid_values():
    """Show all valid values for each field in HTML format"""
    if model is None:
        return "<html><body><h1>Model not loaded. Please train the model first.</h1></body></html>"
    
    valid_values_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Valid Values for Accident Prediction System</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 900px;
                margin: 0 auto;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            }
            .container {
                background-color: white;
                padding: 30px;
                border-radius: 15px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            }
            h1 {
                color: #333;
                text-align: center;
                margin-bottom: 30px;
            }
            h2 {
                color: #667eea;
                border-bottom: 2px solid #667eea;
                padding-bottom: 10px;
                margin-top: 30px;
            }
            ul {
                list-style-type: none;
                padding: 0;
            }
            li {
                background-color: #f5f5f5;
                margin: 8px 0;
                padding: 10px;
                border-radius: 5px;
                font-family: monospace;
            }
            .badge {
                background-color: #4CAF50;
                color: white;
                padding: 3px 8px;
                border-radius: 3px;
                font-size: 12px;
                margin-left: 10px;
            }
            .numeric {
                background-color: #2196F3;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔍 Valid Values for Accident Prediction System</h1>
    """
    
    for col, encoder in label_encoders.items():
        valid_values_html += f"""
            <h2>{col} <span class="badge">{len(encoder.classes_)} values</span></h2>
            <ul>
        """
        for val in encoder.classes_:
            valid_values_html += f"<li>• {val}</li>"
        valid_values_html += "</ul>"
    
    # Add numeric columns info
    numeric_cols = ['Speed_limit', 'Number_of_Vehicles', 'Number_of_Casualties']
    valid_values_html += "<h2>Numeric Fields</h2><ul>"
    for col in numeric_cols:
        if col in features:
            valid_values_html += f"<li>• {col}: Any integer value (e.g., 30, 2, 1)</li>"
    valid_values_html += "</ul>"
    
    valid_values_html += """
            <div style="text-align: center; margin-top: 30px;">
                <button onclick="window.location.href='/'" style="padding: 10px 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; border-radius: 5px; cursor: pointer; font-size: 16px;">
                    Back to Prediction Form
                </button>
            </div>
        </div>
    </body>
    </html>
    """
    
    return valid_values_html

@app.route('/dynamic_form')
def dynamic_form():
    """Dynamic form that loads valid values directly from the model"""
    if model is None:
        return "<html><body><h1>Model not loaded. Please train the model first by running: python model/train.py</h1></body></html>"
    
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
            body {
                font-family: Arial, sans-serif;
                max-width: 900px;
                margin: 0 auto;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }
            .container {
                background-color: white;
                padding: 30px;
                border-radius: 15px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            }
            h1 {
                color: #333;
                text-align: center;
                margin-bottom: 10px;
            }
            .subtitle {
                text-align: center;
                color: #666;
                margin-bottom: 30px;
            }
            .form-group {
                margin-bottom: 20px;
            }
            label {
                display: block;
                margin-bottom: 8px;
                font-weight: bold;
                color: #555;
                font-size: 14px;
            }
            select, input {
                width: 100%;
                padding: 10px;
                border: 2px solid #ddd;
                border-radius: 8px;
                font-size: 14px;
                transition: border-color 0.3s;
                box-sizing: border-box;
            }
            select:focus, input:focus {
                outline: none;
                border-color: #667eea;
            }
            button {
                width: 100%;
                padding: 12px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                border-radius: 8px;
                cursor: pointer;
                font-size: 16px;
                font-weight: bold;
                margin-top: 10px;
                transition: transform 0.2s;
            }
            button:hover {
                transform: translateY(-2px);
            }
            .row {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
            }
            h3 {
                color: #667eea;
                margin-top: 20px;
                margin-bottom: 15px;
            }
            hr {
                margin: 20px 0;
                border: none;
                border-top: 2px solid #f0f0f0;
            }
            .info-note {
                background-color: #f0f0f0;
                padding: 10px;
                border-radius: 5px;
                font-size: 12px;
                color: #666;
                margin-top: 20px;
                text-align: center;
            }
            .accuracy-badge {
                background-color: #4CAF50;
                color: white;
                padding: 5px 10px;
                border-radius: 20px;
                font-size: 12px;
                display: inline-block;
                margin-bottom: 20px;
            }
            .nav-links {
                text-align: center;
                margin-bottom: 20px;
            }
            .nav-links a {
                color: #667eea;
                margin: 0 10px;
                text-decoration: none;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚗 Smart Traffic Prediction & Accident Risk System</h1>
            <div class="subtitle">Dynamic Form - Automatically loads valid values from model</div>
            <div style="text-align: center;">
                <span class="accuracy-badge">Model Accuracy: 85.5%</span>
            </div>
            <div class="nav-links">
                <a href="/valid_values">View All Valid Values</a> | 
                <a href="/debug">Debug API</a>
            </div>
            
            <form action="/predict" method="post">
                <h3>📅 Time & Location</h3>
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
                
                <h3>🛣️ Road Conditions</h3>
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
                        <input type="number" name="Speed_limit" value="30" min="10" max="70" step="1" required>
                    </div>
                </div>
                
                <h3>☁️ Environmental Conditions</h3>
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
                
                <h3>🚑 Accident Details</h3>
                <div class="row">
                    <div class="form-group">
                        <label>Number of Vehicles Involved:</label>
                        <input type="number" name="Number_of_Vehicles" min="1" max="10" value="2" required>
                    </div>
                    <div class="form-group">
                        <label>Number of Casualties:</label>
                        <input type="number" name="Number_of_Casualties" min="0" max="20" value="1" required>
                    </div>
                </div>
                
                <hr>
                <button type="submit">🔍 Predict Accident Severity</button>
            </form>
            <div class="info-note">
                ℹ️ Based on historical accident data from 307,973 incidents | Model accuracy: 85.5%<br>
                ⚠️ This form automatically loads all valid values from the trained model
            </div>
        </div>
    </body>
    </html>
    ''', field_values=field_values)

# Add this import at the top if not already there
from flask import render_template_string

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=True)