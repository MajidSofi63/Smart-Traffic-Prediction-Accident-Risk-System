from flask import Flask, render_template, request, jsonify
import requests
import os
import sib_api_v3_sdk
from datetime import datetime

app = Flask(__name__)

# API Keys
TOMTOM_KEY = os.environ.get('TOMTOM_API_KEY', '')
BREVO_KEY = os.environ.get('BREVO_API_KEY', '')
BREVO_EMAIL = os.environ.get('BREVO_SENDER_EMAIL', 'majidsofi63@gmail.com')

# Sample accident hotspots for heatmap (lat, lng, intensity)
ACCIDENT_HOTSPOTS = [
    # High risk areas (Red)
    [28.6139, 77.2090, 0.9],   # Delhi - Connaught Place
    [28.6300, 77.2200, 0.95],  # ITO intersection
    [28.5600, 77.3200, 0.85],  # Noida
    [19.0760, 72.8777, 0.85],  # Mumbai - Bandra
    [19.1000, 72.8500, 0.8],   # Andheri
    [12.9716, 77.5946, 0.8],   # Bangalore
    [22.5726, 88.3639, 0.75],  # Kolkata
    
    # Medium risk areas (Yellow)
    [28.5000, 77.3000, 0.6],   # Yamuna Expressway
    [28.4000, 77.2000, 0.65],  # Gurgaon
    [13.0827, 80.2707, 0.6],   # Chennai
    [17.3850, 78.4867, 0.55],  # Hyderabad
    [26.9124, 75.7873, 0.5],   # Jaipur
    
    # Low risk areas (Green)
    [26.8000, 78.0000, 0.2],   # Rural UP
    [21.0000, 75.0000, 0.15],  # Rural MP
    [18.0000, 76.0000, 0.1],   # Rural Maharashtra
]

def geocode(address):
    """Convert address to coordinates"""
    if not address:
        return None
    
    # Check if already coordinates
    if ',' in address and address.replace('.', '').replace('-', '').replace(',', '').strip().isdigit():
        parts = address.split(',')
        return {'lat': float(parts[0]), 'lon': float(parts[1])}
    
    url = f"https://api.tomtom.com/search/2/geocode/{address}.json"
    params = {'key': TOMTOM_KEY, 'limit': 1}
    
    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        if data.get('results'):
            pos = data['results'][0]['position']
            return {'lat': pos['lat'], 'lon': pos['lon']}
    except:
        pass
    return None

def calculate_risk(distance_km, duration_min, traffic_delay_min, vehicle, time_of_day, weather):
    """Calculate risk score based on conditions"""
    risk_score = 0
    factors = []
    
    # Distance risk
    if distance_km > 500:
        risk_score += 30
        factors.append(f"Very long distance ({distance_km:.0f} km)")
    elif distance_km > 200:
        risk_score += 20
        factors.append(f"Long distance ({distance_km:.0f} km)")
    elif distance_km > 100:
        risk_score += 10
        factors.append(f"Moderate distance ({distance_km:.0f} km)")
    
    # Traffic delay risk
    if traffic_delay_min > 30:
        risk_score += 35
        factors.append(f"Heavy traffic delay ({traffic_delay_min:.0f} min)")
    elif traffic_delay_min > 15:
        risk_score += 20
        factors.append(f"Moderate traffic delay ({traffic_delay_min:.0f} min)")
    elif traffic_delay_min > 5:
        risk_score += 10
        factors.append(f"Light traffic delay ({traffic_delay_min:.0f} min)")
    
    # Time of day risk
    if time_of_day == 'Night':
        risk_score += 30
        factors.append("Night travel - 3x higher fatality rate")
    elif time_of_day == 'Evening':
        risk_score += 15
        factors.append("Evening travel - peak accident hours")
    elif time_of_day == 'Afternoon':
        risk_score += 5
    
    # Weather risk
    if weather == 'Rain':
        risk_score += 25
        factors.append("Rain - reduced visibility and traction")
    elif weather == 'Fog':
        risk_score += 30
        factors.append("Fog - very poor visibility")
    elif weather == 'Snow':
        risk_score += 35
        factors.append("Snow - hazardous road conditions")
    
    # Vehicle risk
    if vehicle == 'Motorcycle':
        risk_score += 25
        factors.append("Motorcycle - most vulnerable vehicle")
    elif vehicle == 'Truck':
        risk_score += 15
        factors.append("Truck - longer stopping distance")
    
    # Determine risk level
    if risk_score >= 60:
        risk_level = 'HIGH'
        severity = 'Serious'
        color = 'red'
    elif risk_score >= 30:
        risk_level = 'MEDIUM'
        severity = 'Slight'
        color = 'orange'
    else:
        risk_level = 'LOW'
        severity = 'Slight'
        color = 'green'
    
    confidence = min(85 + (risk_score // 10), 98)
    
    return {
        'level': risk_level,
        'severity': severity,
        'confidence': confidence,
        'score': risk_score,
        'color': color,
        'factors': factors[:5]
    }

def get_route(start_addr, end_addr, vehicle, time_of_day, weather):
    """Get route from TomTom API"""
    start = geocode(start_addr)
    end = geocode(end_addr)
    
    if not start or not end:
        return None
    
    url = f"https://api.tomtom.com/routing/1/calculateRoute/{start['lat']},{start['lon']}:{end['lat']},{end['lon']}/json"
    params = {
        'key': TOMTOM_KEY,
        'traffic': 'true',
        'travelMode': 'car',
        'routeType': 'fastest'
    }
    
    try:
        response = requests.get(url, params=params, timeout=15)
        data = response.json()
        
        if not data.get('routes'):
            return None
        
        route = data['routes'][0]
        leg = route['legs'][0]
        summary = leg['summary']
        
        # Extract points for map
        points = []
        for point in leg['points']:
            points.append([point['latitude'], point['longitude']])
        
        distance_km = summary['lengthInMeters'] / 1000
        duration_min = summary['travelTimeInSeconds'] / 60
        traffic_delay = summary.get('trafficDelayInSeconds', 0) / 60
        
        # Calculate risk
        risk = calculate_risk(distance_km, duration_min, traffic_delay, vehicle, time_of_day, weather)
        
        return {
            'distance_km': distance_km,
            'duration_min': duration_min,
            'traffic_delay_min': traffic_delay,
            'points': points,
            'risk': risk,
            'start': {'lat': start['lat'], 'lon': start['lon']},
            'end': {'lat': end['lat'], 'lon': end['lon']}
        }
    except Exception as e:
        print(f"Route error: {e}")
        return None

def send_email(to_email, route):
    """Send email alert"""
    if not BREVO_KEY or not to_email:
        return False
    
    try:
        config = sib_api_v3_sdk.Configuration()
        config.api_key['api-key'] = BREVO_KEY
        api = sib_api_v3_sdk.TransactionalEmailsApi(sib_api_v3_sdk.ApiClient(config))
        
        r = route['risk']
        html = f"""
        <html>
        <body style="font-family: Arial;">
            <h2>🚗 Trip Risk Assessment</h2>
            <div style="background:{r['color']}; color:white; padding:15px; text-align:center; border-radius:10px;">
                <h3>{r['level']} RISK</h3>
                <p>{r['severity']} Accident ({r['confidence']}% confidence)</p>
            </div>
            <div style="margin:15px 0; padding:10px; background:#f0f0f0; border-radius:5px;">
                <p>📍 Distance: {route['distance_km']:.0f} km</p>
                <p>⏱️ Duration: {route['duration_min']:.0f} minutes</p>
                <p>🚦 Traffic Delay: {route['traffic_delay_min']:.0f} minutes</p>
            </div>
            <h3>⚠️ Risk Factors:</h3>
            <ul>
        """
        for f in r['factors']:
            html += f"<li>{f}</li>"
        html += """
            </ul>
            <p>💡 Drive safely! 🚗</p>
        </body>
        </html>
        """
        
        sender = {"name": "Traffic Alert", "email": BREVO_EMAIL}
        to = [{"email": to_email}]
        
        api.send_transac_email(sib_api_v3_sdk.SendSmtpEmail(
            to=to, sender=sender, 
            subject=f"🚨 Trip Risk: {r['level']} - {r['severity']} Accident Possible",
            html_content=html
        ))
        return True
    except Exception as e:
        print(f"Email error: {e}")
        return False

# ============ FLASK ROUTES ============
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/map')
def map_page():
    return render_template('map_route.html')

@app.route('/api/route', methods=['POST'])
def api_route():
    data = request.json
    result = get_route(
        data.get('start'),
        data.get('end'),
        data.get('vehicle', 'Car'),
        data.get('time', 'Morning'),
        data.get('weather', 'Clear')
    )
    
    if not result:
        return jsonify({'error': 'Route not found'}), 400
    
    email_sent = False
    if data.get('email'):
        email_sent = send_email(data['email'], result)
    
    return jsonify({
        'route': result,
        'email_sent': email_sent,
        'hotspots': ACCIDENT_HOTSPOTS
    })

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'api': 'TomTom'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)