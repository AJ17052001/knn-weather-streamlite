# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

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

# New weather data
new_weather = np.array([[26, 78]])

# Create KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

# Predict weather
pred = knn.predict(new_weather)[0]

# Label mapping
label_map = {0: "Sunny", 1: "Rainy"}

# Plot Sunny points
plt.scatter(
    X[y == 0, 0], X[y == 0, 1],
    color='orange', label='Sunny', s=100, edgecolor='k'
)

# Plot Rainy points
plt.scatter(
    X[y == 1, 0], X[y == 1, 1],
    color='blue', label='Rainy', s=100, edgecolor='k'
)

# Plot new weather point
colors = ['orange', 'blue']
plt.scatter(
    new_weather[0, 0], new_weather[0, 1],
    color=colors[pred], marker='*',
    s=300, edgecolor='black',
    label=f'New Day: {label_map[pred]}'
)

# Add prediction text
plt.text(
    new_weather[0, 0] + 0.5, new_weather[0, 1],
    f'Predicted: {label_map[pred]}',
    fontsize=12, color=colors[pred]
)

# Labels and title
plt.xlabel('Temperature')
plt.ylabel('Humidity')
plt.title('KNN Weather Classification')
plt.legend()
plt.grid(True)
plt.show()
