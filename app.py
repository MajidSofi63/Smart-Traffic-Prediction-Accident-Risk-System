from flask import Flask, render_template, request, jsonify
import requests
import os
import sib_api_v3_sdk

app = Flask(__name__)

# ============ API KEYS ============
TOMTOM_KEY = os.environ.get('TOMTOM_API_KEY', 'YOUR_TOMTOM_KEY_HERE')
BREVO_KEY = os.environ.get('BREVO_API_KEY', '')
BREVO_EMAIL = os.environ.get('BREVO_SENDER_EMAIL', 'majidsofi63@gmail.com')

# ============ TOMTOM GEOCODING (Address → Coordinates) ============
def geocode(address):
    """Convert any address (street, city, landmark) to coordinates using TomTom."""
    if not address:
        return None
    
    # If already coordinates
    if ',' in address:
        parts = address.split(',')
        try:
            lat = float(parts[0].strip())
            lon = float(parts[1].strip())
            return {'lat': lat, 'lon': lon}
        except:
            pass
    
    url = "https://api.tomtom.com/search/2/geocode/{}.json".format(requests.utils.quote(address))
    params = {
        'key': TOMTOM_KEY,
        'limit': 1,
        'countrySet': 'IN'  # Limit to India for better accuracy
    }
    
    try:
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        if data.get('results'):
            pos = data['results'][0]['position']
            return {'lat': pos['lat'], 'lon': pos['lon']}
    except Exception as e:
        print(f"Geocode error: {e}")
    return None

# ============ TOMTOM ROUTING (with live traffic) ============
def get_route(start_coords, end_coords):
    """Get route with live traffic from TomTom."""
    url = f"https://api.tomtom.com/routing/1/calculateRoute/{start_coords['lat']},{start_coords['lon']}:{end_coords['lat']},{end_coords['lon']}/json"
    params = {
        'key': TOMTOM_KEY,
        'traffic': 'true',          # Live traffic
        'travelMode': 'car',
        'routeType': 'fastest',
        'computeTravelTimeFor': 'all'
    }
    
    try:
        resp = requests.get(url, params=params, timeout=15)
        data = resp.json()
        if not data.get('routes'):
            return None
        
        route = data['routes'][0]
        leg = route['legs'][0]
        points = []
        for point in leg['points']:
            points.append([point['latitude'], point['longitude']])
        
        summary = leg['summary']
        return {
            'distance_km': summary['lengthInMeters'] / 1000,
            'duration_min': summary['travelTimeInSeconds'] / 60,
            'traffic_delay_min': summary.get('trafficDelayInSeconds', 0) / 60,
            'points': points
        }
    except Exception as e:
        print(f"Route error: {e}")
        return None

# ============ RISK CALCULATION (same as before) ============
def calculate_risk(distance_km, duration_min, vehicle, time_of_day, weather):
    risk_score = 0
    factors = []
    
    if distance_km > 500:
        risk_score += 35
        factors.append(f"Very long distance ({distance_km:.0f} km)")
    elif distance_km > 200:
        risk_score += 20
        factors.append(f"Long distance ({distance_km:.0f} km)")
    elif distance_km > 100:
        risk_score += 10
    
    if time_of_day == 'Night':
        risk_score += 35
        factors.append("🌙 Night travel – 3x higher fatality rate")
    elif time_of_day == 'Evening':
        risk_score += 20
        factors.append("🌆 Evening travel – peak accident hours")
    
    if weather == 'Rain':
        risk_score += 25
        factors.append("🌧️ Rain – reduced visibility")
    elif weather == 'Fog':
        risk_score += 30
        factors.append("🌫️ Fog – very poor visibility")
    
    if vehicle == 'Motorcycle':
        risk_score += 25
        factors.append("🏍️ Motorcycle – most vulnerable")
    elif vehicle == 'Truck':
        risk_score += 15
        factors.append("🚛 Truck – longer stopping distance")
    
    level = 'HIGH' if risk_score >= 60 else 'MEDIUM' if risk_score >= 30 else 'LOW'
    return {'level': level, 'score': risk_score, 'factors': factors[:5]}

# ============ EMAIL ALERT (Brevo) ============
def send_email(to_email, route_info, risk):
    if not BREVO_KEY or not to_email:
        return False
    try:
        config = sib_api_v3_sdk.Configuration()
        config.api_key['api-key'] = BREVO_KEY
        api = sib_api_v3_sdk.TransactionalEmailsApi(sib_api_v3_sdk.ApiClient(config))
        
        color = {'HIGH':'red','MEDIUM':'orange','LOW':'green'}.get(risk['level'],'gray')
        html = f"""
        <html><body>
        <h2>🚗 Route Risk Assessment</h2>
        <div style="background:{color};color:white;padding:15px;text-align:center">
            <h3>{risk['level']} RISK (Score: {risk['score']}/100)</h3>
        </div>
        <p>📍 Distance: {route_info['distance_km']:.0f} km<br>⏱️ Duration: {route_info['duration_min']:.0f} min</p>
        <h3>⚠️ Risk Factors</h3><ul>"""
        for f in risk['factors']:
            html += f"<li>{f}</li>"
        html += "</ul><p>Stay safe! 🚗</p></body></html>"
        
        sender = {"name":"Traffic Alert","email":BREVO_EMAIL}
        to = [{"email":to_email}]
        api.send_transac_email(sib_api_v3_sdk.SendSmtpEmail(
            to=to, sender=sender,
            subject=f"Route Risk: {risk['level']}",
            html_content=html
        ))
        return True
    except Exception as e:
        print(f"Email error: {e}")
        return False

# ============ FLASK ROUTES ============
@app.route('/')
def home():
    return render_template('map_route.html')

@app.route('/map')
def map_page():
    return render_template('map_route.html')

@app.route('/api/route', methods=['POST'])
def api_route():
    data = request.json
    start = data.get('start')
    end = data.get('end')
    vehicle = data.get('vehicle', 'Car')
    time_of_day = data.get('time', 'Morning')
    weather = data.get('weather', 'Clear')
    user_email = data.get('email', '')
    
    if not start or not end:
        return jsonify({'error': 'Missing start or destination'}), 400
    
    start_coords = geocode(start)
    end_coords = geocode(end)
    if not start_coords or not end_coords:
        return jsonify({'error': 'Could not find addresses. Try being more specific.'}), 400
    
    route = get_route(start_coords, end_coords)
    if not route:
        return jsonify({'error': 'Could not calculate route'}), 400
    
    risk = calculate_risk(route['distance_km'], route['duration_min'], vehicle, time_of_day, weather)
    email_sent = send_email(user_email, route, risk) if user_email else False
    
    return jsonify({
        'route': {
            'distance_km': route['distance_km'],
            'duration_min': route['duration_min'],
            'traffic_delay_min': route['traffic_delay_min'],
            'points': route['points']
        },
        'risk': risk,
        'email_sent': email_sent
    })

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'geocoding': 'TomTom', 'routing': 'TomTom'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)