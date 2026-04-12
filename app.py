from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model = pickle.load(open("model/traffic_model.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    time = int(request.form['time'])
    weather = request.form['weather']
    vehicles = int(request.form['vehicles'])

    weather_map = {'clear':0,'rain':1,'fog':2}
    pred = model.predict([[time, weather_map[weather], vehicles]])

    traffic_map = {0:"Low",1:"Medium",2:"High"}
    traffic = traffic_map[pred[0]]

    # Accident Risk
    if weather in ["rain","fog"]:
        risk = "High"
    elif time > 22 or time < 5:
        risk = "Medium"
    else:
        risk = "Low"

    return render_template(
    'result.html',
    traffic=traffic,
    risk=risk,
    time=time,
    vehicles=vehicles
)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000)