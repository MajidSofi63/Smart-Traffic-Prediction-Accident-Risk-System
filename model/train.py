import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Create model directory
os.makedirs('model', exist_ok=True)

# Load the data
print("Loading data...")
data = pd.read_csv('data/Road Accident Data.csv')

print(f"Dataset shape: {data.shape}")
print(f"\nAvailable columns: {data.columns.tolist()}")

# Use the ACTUAL column names from your dataset
selected_features = [
    'Day_of_Week',
    'Junction_Control',
    'Light_Conditions',
    'Road_Surface_Conditions',
    'Road_Type',
    'Speed_limit',
    'Urban_or_Rural_Area',
    'Weather_Conditions',
    'Number_of_Vehicles',
    'Number_of_Casualties'
]

target = 'Accident_Severity'

# Check which features actually exist in the dataset
available_features = [f for f in selected_features if f in data.columns]
missing_features = [f for f in selected_features if f not in data.columns]

print(f"\n✓ Available features: {available_features}")
if missing_features:
    print(f"⚠ Missing features: {missing_features}")

# Prepare features and target
X = data[available_features].copy()
y = data[target].copy()

# Handle missing values
print(f"\nMissing values before handling:")
print(f"X missing: {X.isnull().sum().sum()}")
print(f"y missing: {y.isnull().sum()}")

# Fill missing values
X = X.fillna('Unknown')
y = y.fillna('Slight')

# Clean target variable - fix any typos
y = y.replace('Fetal', 'Fatal')

# Convert numeric columns to proper types
numeric_cols = ['Speed_limit', 'Number_of_Vehicles', 'Number_of_Casualties']
for col in numeric_cols:
    if col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0).astype(int)

# Encode all categorical columns
print("\nEncoding categorical variables...")
label_encoders = {}

for column in X.columns:
    if column in numeric_cols:
        print(f"  - {column}: keeping as numeric")
        continue
    
    le = LabelEncoder()
    # Convert to string first to handle any non-string values
    X[column] = X[column].astype(str)
    X[column] = le.fit_transform(X[column])
    label_encoders[column] = le
    print(f"  - {column}: encoded {len(le.classes_)} unique values")

# Encode target variable
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y.astype(str))
print(f"\nTarget classes: {target_encoder.classes_.tolist()}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

# Train the model
print("\nTraining Random Forest Classifier...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
    verbose=1,
    class_weight='balanced'  # This helps with imbalanced classes
)

model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n{'='*50}")
print(f"Model Accuracy: {accuracy:.4f}")
print(f"{'='*50}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_encoder.classes_))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': available_features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance:")
for idx, row in feature_importance.iterrows():
    print(f"  {row['feature']}: {row['importance']:.4f}")

# Save the model and encoders
print("\nSaving model and encoders...")
joblib.dump(model, 'model/traffic_model.pkl')
joblib.dump(label_encoders, 'model/label_encoders.pkl')
joblib.dump(target_encoder, 'model/target_encoder.pkl')
joblib.dump(available_features, 'model/selected_features.pkl')

print("✓ Model saved successfully to 'model/' directory!")

# Test with a sample prediction
print("\n" + "="*50)
print("Testing with sample predictions...")

# Get a few sample rows from the original data for testing
sample_indices = [0, 1, 2, 100, 1000]  # First 5 rows
for idx in sample_indices:
    if idx < len(X):
        sample = X.iloc[[idx]].copy()
        actual_value = y[idx]  # y is numpy array, use indexing not iloc
        
        # Make prediction
        pred = model.predict(sample)[0]
        pred_severity = target_encoder.inverse_transform([pred])[0]
        actual_severity = target_encoder.inverse_transform([actual_value])[0]
        
        print(f"\nSample {idx+1}:")
        print(f"  Actual: {actual_severity}")
        print(f"  Predicted: {pred_severity}")
        print(f"  Correct: {actual_value == pred}")

print("\n✓ Training complete! You can now run: python app.py")