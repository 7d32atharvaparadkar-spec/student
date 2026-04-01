import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Load dataset
df = pd.read_csv("student.csv")

# Features & Target
X = df[["StudyHours","Attendance"]]
y = df["Marks"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
pickle.dump(model, open("student_model.pkl","wb"))

print("✅ Model Trained & Saved")