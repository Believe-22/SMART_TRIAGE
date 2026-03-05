# SMART_TRIAGE
# TRAINING SCRIPT

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Create synthetic dataset
np.random.seed(42)

data_size = 500

hr = np.random.randint(60, 150, data_size)
spo2 = np.random.randint(85, 100, data_size)
temp = np.random.uniform(36, 41, data_size)

# Create labels based on logical rules
labels = []

for i in range(data_size):
    if spo2[i] < 90:
        labels.append("Critical")
    elif temp[i] > 39.5 and hr[i] > 110:
        labels.append("Critical")
    elif hr[i] > 100 or temp[i] > 38 or spo2[i] < 95:
        labels.append("Medium")
    else:
        labels.append("Low")

# Create DataFrame
df = pd.DataFrame({
    "HeartRate": hr,
    "SpO2": spo2,
    "Temperature": temp,
    "Label": labels
})

# Split
X = df[["HeartRate", "SpO2", "Temperature"]]
y = df["Label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))

# Save model
joblib.dump(model, "triage_model.pkl")

print("Model trained and saved.")

# TESTING MODEL
#TRAINED MODEL
import streamlit as st
import joblib
import numpy as np

model = joblib.load("triage_model.pkl")

st.title("SmartTriage-Ug")
st.subheader("AI-Powered Triage Decision Support")

hr = st.number_input("Heart Rate (bpm)", min_value=30, max_value=200, value=80)
spo2 = st.number_input("SpO2 (%)", min_value=50, max_value=100, value=98)
temp = st.number_input("Temperature (°C)", min_value=30.0, max_value=45.0, value=36.8)

if st.button("Analyze"):
    input_data = np.array([[hr, spo2, temp]])
    prediction = model.predict(input_data)[0]
    
  if prediction == "Critical":
        st.error(f"Urgency Level: {prediction} 🔴")
    elif prediction == "Medium":
        st.warning(f"Urgency Level: {prediction} 🟡")
    else:
        st.success(f"Urgency Level: {prediction} 🟢")
