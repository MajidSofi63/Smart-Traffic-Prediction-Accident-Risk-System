import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

data = pd.read_csv("../data/traffic.csv")

data['weather'] = data['weather'].map({'clear':0,'rain':1,'fog':2})
data['traffic'] = data['traffic'].map({'low':0,'medium':1,'high':2})

X = data[['time','weather','vehicles']]
y = data['traffic']

model = RandomForestClassifier()
model.fit(X, y)

pickle.dump(model, open("traffic_model.pkl", "wb"))

print("Model trained successfully!")