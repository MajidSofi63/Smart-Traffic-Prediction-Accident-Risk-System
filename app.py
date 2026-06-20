from flask import Flask, render_template, request, jsonify
import requests
import os
import sib_api_v3_sdk
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# ============ API KEYS (from environment variables) ============
TOMTOM_KEY = os.environ.get('TOMTOM_API_KEY', '')
BREVO_KEY = os.environ.get('BREVO_API_KEY', '')
BREVO_EMAIL = os.environ.get('BREVO_SENDER_EMAIL', '')

# ============ TOMTOM GEOCODING ============
def geocode(address):
    if not address:
        return None
    if ',' in address:
        parts = address.split(',')
        try:
            lat = float(parts[0].strip())
            lon = float(parts[1].strip())
            return {'lat': lat, 'lon': lon}
        except:
            pass
    url = "https://api.tomtom.com/search/2/geocode/{}.json".format(requests.utils.quote(address))
    params = {'key': TOMTOM_KEY, 'limit': 1, 'countrySet': 'IN'}
    try:
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        if data.get('results'):
            pos = data['results'][0]['position']
            return {'lat': pos['lat'], 'lon': pos['lon']}
    except:
        pass
    return None

# ============ TOMTOM ROUTING (with live traffic + guidance) ============
def get_routes(start_coords, end_coords, travel_mode='car'):
    url = f"https://api.tomtom.com/routing/1/calculateRoute/{start_coords['lat']},{start_coords['lon']}:{end_coords['lat']},{end_coords['lon']}/json"
    params = {
        'key': TOMTOM_KEY,
        'traffic': 'true',
        'travelMode': travel_mode,
        'routeType': 'fastest',
        'computeTravelTimeFor': 'all',
        'maxAlternatives': 1,
        'instructionsType': 'text',
        'sectionType': 'traffic',
        'report': 'effectiveSettings'
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        data = resp.json()
        if not data.get('routes'):
            return []

        parsed_routes = []
        for idx, route in enumerate(data['routes']):
            leg = route['legs'][0]
            points = [[p['latitude'], p['longitude']] for p in leg['points']]
            summary = leg['summary']

            # Parse turn-by-turn guidance instructions
            instructions = []
            guidance = route.get('guidance', {})
            for inst in guidance.get('instructions', []):
                instructions.append({
                    'message': inst.get('message', ''),
                    'street': inst.get('street', ''),
                    'maneuver': inst.get('maneuver', 'STRAIGHT'),
                    'point_index': inst.get('pointIndex', 0),
                    'route_offset_m': inst.get('routeOffsetInMeters', 0)
                })

            parsed_routes.append({
                'route_index': idx,
                'distance_km': summary['lengthInMeters'] / 1000,
                'duration_min': summary['travelTimeInSeconds'] / 60,
                'traffic_delay_min': summary.get('trafficDelayInSeconds', 0) / 60,
                'points': points,
                'instructions': instructions
            })
        return parsed_routes
    except Exception as e:
        print(f"Error fetching routes: {e}")
        return []

# ============ WEATHER (Open‑Meteo, no API key) ============
def get_weather(lat, lon):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        w = data.get('current_weather', {})
        code = w.get('weathercode')
        weather_map = {
            0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
            45: "Fog", 51: "Light drizzle", 61: "Rain", 71: "Snow"
        }
        desc = weather_map.get(code, "Unknown")
        return {
            'temp': w.get('temperature'),
            'wind': w.get('windspeed'),
            'desc': desc
        }
    except:
        return None

# ============ LOAD ML MODEL AND ENCODERS ============
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')
try:
    traffic_model = joblib.load(os.path.join(MODEL_DIR, 'traffic_model.pkl'))
    label_encoders = joblib.load(os.path.join(MODEL_DIR, 'label_encoders.pkl'))
    target_encoder = joblib.load(os.path.join(MODEL_DIR, 'target_encoder.pkl'))
    selected_features = joblib.load(os.path.join(MODEL_DIR, 'selected_features.pkl'))
    model_loaded = True
    print("ML model and encoders loaded successfully.")
except Exception as e:
    traffic_model = None
    label_encoders = {}
    target_encoder = None
    selected_features = []
    model_loaded = False
    print(f"Failed to load ML model or encoders: {e}")

def fallback_risk_calculation(inputs_dict):
    risk_score = 0
    factors = []
    
    speed = int(inputs_dict.get('Speed_limit', 30))
    if speed >= 60:
        risk_score += 30
        factors.append("⚡ High speed limit zone increases risks")
    elif speed >= 40:
        risk_score += 15
        
    weather = inputs_dict.get('Weather_Conditions', 'Fine no high winds')
    if 'Rain' in weather or 'Wet' in inputs_dict.get('Road_Surface_Conditions', ''):
        risk_score += 20
        factors.append("🌧️ Wet road surface reduces traction")
        
    light = inputs_dict.get('Light_Conditions', 'Daylight')
    if 'Darkness' in light:
        risk_score += 25
        factors.append("🌙 Night travel visibility hazard")
        
    urban_rural = inputs_dict.get('Urban_or_Rural_Area', 'Urban')
    if urban_rural == 'Rural':
        risk_score += 15
        factors.append("🏡 Rural area emergency response latency")
        
    level = 'HIGH' if risk_score >= 50 else 'MEDIUM' if risk_score >= 25 else 'LOW'
    return {
        'level': level,
        'score': risk_score,
        'severity': 'Slight' if level == 'LOW' else 'Serious' if level == 'MEDIUM' else 'Fatal',
        'confidence': 70,
        'factors': factors if factors else ["Standard route conditions"]
    }

# ============ RISK CALCULATION ============
def predict_accident_risk(inputs_dict, vehicle_type='Car'):
    if not model_loaded:
        return fallback_risk_calculation(inputs_dict)

    try:
        df = pd.DataFrame([inputs_dict])
        
        # Verify columns and apply label encoding
        for col in selected_features:
            if col not in df.columns:
                if col in ['Speed_limit', 'Number_of_Vehicles', 'Number_of_Casualties']:
                    df[col] = 30
                else:
                    df[col] = 'Unknown'
            
            if col in label_encoders:
                le = label_encoders[col]
                val = str(df[col].iloc[0])
                
                # Check for typo match or closest category
                if val not in le.classes_:
                    matching_class = [c for c in le.classes_ if val.lower() in c.lower()]
                    if matching_class:
                        val = matching_class[0]
                    else:
                        val = le.classes_[0]
                
                df[col] = le.transform([val])
            else:
                df[col] = pd.to_numeric(df[col]).fillna(0).astype(int)
        
        X_pred = df[selected_features]
        
        pred_class = traffic_model.predict(X_pred)[0]
        pred_proba = traffic_model.predict_proba(X_pred)[0]
        
        classes = list(target_encoder.classes_)
        p_fatal = pred_proba[classes.index('Fatal')] if 'Fatal' in classes else 0.0
        p_serious = pred_proba[classes.index('Serious')] if 'Serious' in classes else 0.0
        p_slight = pred_proba[classes.index('Slight')] if 'Slight' in classes else 0.0
        
        # Composite risk score calculation (0 - 100)
        risk_score = (p_fatal * 100) + (p_serious * 60) + (p_slight * 20)
        
        # Vehicle modifiers
        if vehicle_type == 'Motorcycle':
            risk_score += 12
        elif vehicle_type == 'Truck':
            risk_score += 5
            
        risk_score = min(max(int(round(risk_score)), 0), 100)
        
        # Determine risk level
        if risk_score >= 60:
            level = 'HIGH'
        elif risk_score >= 30:
            level = 'MEDIUM'
        else:
            level = 'LOW'
            
        severity = target_encoder.inverse_transform([pred_class])[0]
        confidence = int(round(pred_proba[pred_class] * 100))
        
        # Derive risk factors based on inputs and probabilities
        factors = []
        if vehicle_type == 'Motorcycle':
            factors.append("🏍️ Motorcycle – highly vulnerable vehicle type with elevated safety hazard.")
        elif vehicle_type == 'Truck':
            factors.append("🚛 Truck – longer stopping distances and heavy vehicle handling complexity.")
            
        if inputs_dict.get('Light_Conditions') in ['Darkness - no lighting', 'Darkness - lights unlit']:
            factors.append("🌙 Unlit or dark road conditions – significantly increases night hazard.")
        if int(inputs_dict.get('Speed_limit', 30)) >= 60:
            factors.append(f"⚡ High speed limit zone ({inputs_dict.get('Speed_limit')} mph) – increases impact forces.")
        if inputs_dict.get('Road_Surface_Conditions') in ['Wet or damp', 'Frost or ice', 'Snow', 'Flood over 3cm. deep']:
            factors.append(f"🌧️ Road surface is slippery ({inputs_dict.get('Road_Surface_Conditions')}) – reduced tyre traction.")
        if inputs_dict.get('Weather_Conditions') in ['Fog or mist', 'Raining + high winds', 'Snowing + high winds']:
            factors.append(f"🌫️ Adverse weather conditions ({inputs_dict.get('Weather_Conditions')}) – severely restricted visibility.")
        if inputs_dict.get('Junction_Control') in ['Give way or uncontrolled']:
            factors.append("🔀 Uncontrolled junction area – high potential for merging conflicts.")
        if inputs_dict.get('Urban_or_Rural_Area') == 'Rural':
            factors.append("🏡 Rural road type – remote response location and high severity index.")
        if int(inputs_dict.get('Number_of_Vehicles', 1)) >= 3:
            factors.append(f"🚗 Multi-vehicle corridor ({inputs_dict.get('Number_of_Vehicles')} vehicles) – elevated risk of chain collision.")
            
        if not factors:
            if level == 'HIGH':
                factors.append("⚠️ Combined road environmental factors indicate an elevated hazard profile.")
            elif level == 'MEDIUM':
                factors.append("ℹ️ Standard suburban road hazards detected along the route.")
            else:
                factors.append("✅ Clear weather and daylight conditions indicate a safe journey.")
                
        return {
            'level': level,
            'score': risk_score,
            'severity': severity,
            'confidence': confidence,
            'factors': factors[:5]
        }
    except Exception as e:
        print(f"Error during ML prediction: {e}")
        return fallback_risk_calculation(inputs_dict)

# ============ EMAIL ALERT (Brevo) ============
def send_email(to_email, route_info, risk):
    if not BREVO_KEY or not to_email:
        return False
    try:
        config = sib_api_v3_sdk.Configuration()
        config.api_key['api-key'] = BREVO_KEY
        api = sib_api_v3_sdk.TransactionalEmailsApi(sib_api_v3_sdk.ApiClient(config))
        color = {'HIGH':'#ef4444','MEDIUM':'#f59e0b','LOW':'#10b981'}.get(risk['level'],'#64748b')
        html = f"""
        <html><body style="font-family: Arial, sans-serif; color: #333;">
        <h2>🚦 Route Risk Assessment Report</h2>
        <div style="background:{color};color:white;padding:15px;text-align:center;border-radius:8px;">
            <h3>{risk['level']} RISK LEVEL (Score: {risk['score']}/100)</h3>
            <p>Predicted Severity: <strong>{risk.get('severity', 'N/A')}</strong> ({risk.get('confidence', 0)}% confidence)</p>
        </div>
        <p>📍 <strong>Distance:</strong> {route_info['distance_km']:.1f} km<br>⏱️ <strong>Duration:</strong> {route_info['duration_min']:.1f} min</p>
        <h3>⚠️ Primary Risk Factors</h3><ul>
        """ + "".join(f"<li style='margin-bottom:6px;'>{f}</li>" for f in risk['factors']) + """
        </ul>
        <p style="font-size: 12px; color: #777; margin-top: 20px;">Stay safe and drive defensively! Smart Traffic IQ</p>
        </body></html>"""
        sender = {"name":"Smart Traffic IQ","email":BREVO_EMAIL}
        to = [{"email":to_email}]
        api.send_transac_email(sib_api_v3_sdk.SendSmtpEmail(
            to=to, sender=sender,
            subject=f"Traffic Risk Report: {risk['level']} Risk Alert",
            html_content=html
        ))
        return True
    except Exception as e:
        print(f"Error sending email: {e}")
        return False

# ============ FLASK ROUTES ============
@app.route('/')
def home():
    return render_template('map_route.html')

@app.route('/map')
def map_page():
    return render_template('map_route.html')

@app.route('/geocode', methods=['POST'])
def geocode_endpoint():
    data = request.json
    addr = data.get('address')
    if not addr:
        return jsonify({'error': 'No address'}), 400
    coords = geocode(addr)
    if coords:
        return jsonify({'lat': coords['lat'], 'lon': coords['lon']})
    return jsonify({'error': 'Location not found'}), 404

@app.route('/autocomplete', methods=['GET'])
def autocomplete():
    query = request.args.get('query', '')
    if not query or len(query) < 2:
        return jsonify([])
    url = "https://api.tomtom.com/search/2/search/{}.json".format(requests.utils.quote(query))
    params = {
        'key': TOMTOM_KEY,
        'limit': 5,
        'countrySet': 'IN',
        'typeahead': 'true'
    }
    try:
        resp = requests.get(url, params=params, timeout=5)
        data = resp.json()
        suggestions = []
        for result in data.get('results', []):
            addr = result.get('address', {}).get('freeformAddress')
            pos = result.get('position')
            if addr and pos:
                suggestions.append({
                    'display': addr,
                    'lat': pos['lat'],
                    'lon': pos['lon']
                })
        return jsonify(suggestions)
    except Exception as e:
        print(f"Autocomplete error: {e}")
        return jsonify([])

@app.route('/api/route', methods=['POST'])
def api_route():
    data = request.json
    start = data.get('start')
    end = data.get('end')
    vehicle = data.get('vehicle', 'Car')
    time_of_day = data.get('time', 'Morning')
    weather_cond = data.get('weather', 'Clear')
    user_email = data.get('email', '')

    # Fetch advanced parameters if supplied, otherwise auto-derive/sensible default
    speed_limit = data.get('speed_limit')
    junction_control = data.get('junction_control')
    road_type = data.get('road_type')
    road_surface = data.get('road_surface')
    urban_rural = data.get('urban_rural')
    num_vehicles = data.get('num_vehicles')
    num_casualties = data.get('num_casualties')

    if not start or not end:
        return jsonify({'error': 'Missing start/destination'}), 400

    start_coords = geocode(start)
    end_coords = geocode(end)
    if not start_coords or not end_coords:
        return jsonify({'error': 'Address not found'}), 400

    # Map vehicle selection to TomTom travelMode
    tomtom_vehicle_map = {
        'Car': 'car',
        'Motorcycle': 'motorcycle',
        'Truck': 'truck'
    }
    travel_mode = tomtom_vehicle_map.get(vehicle, 'car')

    routes = get_routes(start_coords, end_coords, travel_mode=travel_mode)
    if not routes:
        return jsonify({'error': 'Route not found'}), 400

    weather_start = get_weather(start_coords['lat'], start_coords['lon'])
    weather_end = get_weather(end_coords['lat'], end_coords['lon'])

    # Determine current day of week
    from datetime import datetime
    day_of_week = datetime.now().strftime('%A')

    # Apply defaults/mappings if advanced parameters are not provided by client
    if not speed_limit:
        speed_limit = 60 if vehicle == 'Truck' else 30
    if not junction_control:
        junction_control = 'Not at junction or within 20 metres'
    if not road_type:
        road_type = 'Single carriageway'
    if not urban_rural:
        urban_rural = 'Urban'
    if not num_vehicles:
        num_vehicles = 2 if vehicle in ['Truck', 'Motorcycle'] else 1
    if not num_casualties:
        num_casualties = 1

    # Map Light conditions from time_of_day
    light_cond = 'Daylight'
    if time_of_day == 'Night':
        light_cond = 'Darkness - no lighting'
    elif time_of_day == 'Evening':
        light_cond = 'Darkness - lights lit'
    elif time_of_day == 'Afternoon':
        light_cond = 'Daylight'

    # Map Road Surface Conditions from weather_cond
    if not road_surface:
        if weather_cond == 'Rain':
            road_surface = 'Wet or damp'
        elif weather_cond == 'Snow':
            road_surface = 'Snow'
        elif weather_cond == 'Fog':
            road_surface = 'Wet or damp'
        else:
            road_surface = 'Dry'

    # Map Weather Conditions for ML model
    weather_ml = 'Fine no high winds'
    if weather_cond == 'Rain':
        weather_ml = 'Raining no high winds'
    elif weather_cond == 'Fog':
        weather_ml = 'Fog or mist'
    elif weather_cond == 'Snow':
        weather_ml = 'Snowing no high winds'

    evaluated_routes = []
    for idx, r in enumerate(routes):
        inputs_dict = {
            'Day_of_Week': day_of_week,
            'Junction_Control': junction_control,
            'Light_Conditions': light_cond,
            'Road_Surface_Conditions': road_surface,
            'Road_Type': road_type,
            'Speed_limit': int(speed_limit),
            'Urban_or_Rural_Area': urban_rural,
            'Weather_Conditions': weather_ml,
            'Number_of_Vehicles': int(num_vehicles),
            'Number_of_Casualties': int(num_casualties)
        }

        # Calculate ML risk
        risk = predict_accident_risk(inputs_dict, vehicle_type=vehicle)
        
        # Differentiate alternative route factors if longer
        if idx > 0 and r['distance_km'] > routes[0]['distance_km']:
            risk['score'] = min(risk['score'] + 5, 100)
            if risk['score'] >= 60:
                risk['level'] = 'HIGH'
            elif risk['score'] >= 30:
                risk['level'] = 'MEDIUM'
            risk['factors'].append("🛣️ Alternative path – extended exposure distance increases overall risk profile.")

        r['risk'] = risk
        evaluated_routes.append(r)

    # Safest route is the one with the lowest risk score
    safest_route = min(evaluated_routes, key=lambda x: x['risk']['score'])
    email_sent = send_email(user_email, safest_route, safest_route['risk']) if user_email else False

    return jsonify({
        'routes': evaluated_routes,
        'email_sent': email_sent,
        'weather_start': weather_start,
        'weather_end': weather_end
    })

@app.route('/api/incidents', methods=['POST'])
def api_incidents():
    data = request.json
    bbox = data.get('bbox')  # [minLat, minLon, maxLat, maxLon]
    if not bbox or not TOMTOM_KEY:
        return jsonify({'incidents': []})

    # TomTom Traffic Incidents API v5: bbox is minLon,minLat,maxLon,maxLat
    bbox_str = f"{bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]}"
    url = "https://api.tomtom.com/traffic/services/5/incidentDetails"
    params = {
        'key': TOMTOM_KEY,
        'bbox': bbox_str,
        'fields': '{incidents{type,geometry{type,coordinates},properties{iconCategory,magnitudeOfDelay,events{description,code},from,to,delay,roadNumbers}}}',
        'language': 'en-GB',
        'categoryFilter': '0,1,2,3,4,5,6,7,8',
        'timeValidityFilter': 'present'
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        result = resp.json()
        incidents = []
        for inc in result.get('incidents', []):
            geom = inc.get('geometry', {})
            props = inc.get('properties', {})
            coords = geom.get('coordinates', [])

            if geom.get('type') == 'Point' and coords:
                point = [coords[1], coords[0]]
            elif geom.get('type') == 'LineString' and coords:
                mid = coords[len(coords)//2]
                point = [mid[1], mid[0]]
            else:
                continue

            events = props.get('events', [])
            desc = events[0].get('description', 'Traffic incident') if events else 'Traffic incident'
            category = props.get('iconCategory', 0)
            delay = props.get('delay', 0)

            incidents.append({
                'lat': point[0],
                'lon': point[1],
                'category': category,
                'description': desc,
                'delay_sec': delay,
                'from': props.get('from', ''),
                'to': props.get('to', '')
            })
        return jsonify({'incidents': incidents})
    except Exception as e:
        print(f"Incidents API error: {e}")
        return jsonify({'incidents': []})

@app.route('/api/reverse_geocode', methods=['GET'])
def reverse_geocode():
    lat = request.args.get('lat')
    lon = request.args.get('lon')
    if not lat or not lon:
        return jsonify({'address': ''})
    url = f"https://api.tomtom.com/search/2/reverseGeocode/{lat},{lon}.json"
    params = {'key': TOMTOM_KEY}
    try:
        resp = requests.get(url, params=params, timeout=8)
        data = resp.json()
        results = data.get('addresses', [])
        if results:
            addr = results[0].get('address', {}).get('freeformAddress', '')
            return jsonify({'address': addr})
    except Exception as e:
        print(f"Reverse geocode error: {e}")
    return jsonify({'address': ''})

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'geocoding': 'TomTom', 'routing': 'TomTom', 'weather': 'Open-Meteo', 'ml': model_loaded})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)