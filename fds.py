# FDS EXP - Library Seat Availability Prediction

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# ---------------------------
# Load Dataset (SAFE PATH)
# ---------------------------

file_path = r"D:\sem-4\fds\library\library_seat_dataset.csv"
data = pd.read_csv(file_path)

print("Dataset Overview:")
print(data.head())

print("\nDataset Info:")
print(data.info())

# ---------------------------
# Data Preprocessing
# ---------------------------

# Convert Day to numeric
data['Day'] = data['Day'].map({
    'Mon': 0,
    'Tue': 1,
    'Wed': 2,
    'Thu': 3,
    'Fri': 4
})

# Features and Target
X = data[['Hour', 'Day', 'Total_Seats']]
y = data['Occupied_Seats']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------
# Model Training
# ---------------------------

model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# ---------------------------
# Evaluation
# ---------------------------

mae = mean_absolute_error(y_test, y_pred)
print("\nMean Absolute Error:", mae)

# ---------------------------
# Visualization
# ---------------------------

# Line Plot
plt.figure()
plt.plot(data['Hour'], data['Occupied_Seats'], marker='o')
plt.title("Library Seat Occupancy by Hour")
plt.xlabel("Hour")
plt.ylabel("Occupied Seats")
plt.grid(True)
plt.show()

# Bar Chart
plt.figure()
data.groupby('Hour')['Occupied_Seats'].mean().plot(kind='bar')
plt.title("Average Occupancy per Hour")
plt.xlabel("Hour")
plt.ylabel("Average Seats Occupied")
plt.show()

# ---------------------------
# Sample Prediction (NO WARNING)
# ---------------------------

sample = pd.DataFrame([[12, 0, 100]],
                      columns=['Hour', 'Day', 'Total_Seats'])

prediction = model.predict(sample)

occupied = int(prediction[0])
available = 100 - occupied

print("\nPredicted Occupied Seats:", occupied)
print("Available Seats:", available)