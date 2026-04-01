import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

st.title("🎓 Student Performance Predictor")

# Load dataset
df = pd.read_csv("student.csv")

# Train model inside app (no error in deployment)
X = df[["StudyHours","Attendance"]]
y = df["Marks"]

model = LinearRegression()
model.fit(X, y)

# Inputs
hours = st.number_input("Study Hours", min_value=0.0)
attendance = st.number_input("Attendance (%)", min_value=0.0, max_value=100.0)

if st.button("Predict Marks"):
    
    data = np.array([[hours, attendance]])
    
    result = model.predict(data)
    
    st.success(f"📊 Predicted Marks = {result[0]:.2f}")