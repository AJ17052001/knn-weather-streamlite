
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# App title
st.title(" KNN Weather Classification")
st.write("Predict weather using Temperature and Humidity")

# Dataset: [Temperature, Humidity]
X = np.array([
    [50, 70],
    [25, 80],
    [27, 60],
    [31, 65],
    [23, 85],
    [20, 75]
])

# Labels: 0 = Sunny, 1 = Rainy
y = np.array([0, 1, 0, 0, 1, 1])

label_map = {0: "Sunny", 1: "Rainy"}

# User input
st.sidebar.header("Enter Weather Details")
temp = st.sidebar.slider("Temperature (°C)", 15, 60, 26)
humidity = st.sidebar.slider("Humidity (%)", 50, 100, 78)

new_weather = np.array([[temp, humidity]])

# Create and train KNN model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

# Prediction
pred = knn.predict(new_weather)[0]

st.subheader(f" Predicted Weather: **{label_map[pred]}**")

# Plot graph
fig, ax = plt.subplots()

# Sunny points
ax.scatter(
    X[y == 0, 0], X[y == 0, 1],
    color='orange', label='Sunny', s=100, edgecolor='k'
)

# Rainy points
ax.scatter(
    X[y == 1, 0], X[y == 1, 1],
    color='blue', label='Rainy', s=100, edgecolor='k'
)

# New weather point
colors = ['orange', 'blue']
ax.scatter(
    temp, humidity,
    color=colors[pred], marker='*',
    s=300, edgecolor='black',
    label=f'New Day: {label_map[pred]}'
)

# Prediction text
ax.text(
    temp + 0.5, humidity,
    f'Predicted: {label_map[pred]}',
    fontsize=12, color=colors[pred]
)

# Labels
ax.set_xlabel("Temperature")
ax.set_ylabel("Humidity")
ax.set_title("KNN Weather Classification")
ax.legend()
ax.grid(True)

# Display plot
st.pyplot(fig)
