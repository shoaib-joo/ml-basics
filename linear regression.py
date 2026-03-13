import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

learning_rate = 0.0001

# Load data

base_dir = os.path.dirname(os.path.abspath(__file__))
data = pd.read_csv(os.path.join(base_dir, "Datasets", "swedish_insurance.csv"))
# Extract X and Y
x = data["X"].to_numpy()
y = data["Y"].to_numpy()

# Augment x with bias term
x_aug = np.column_stack((np.ones(x.shape[0]), x))
print(x_aug)

# Initial weights
w = np.array([5, 10])

epochs = 100
for _ in range(epochs):
    for i in range(x.shape[0]):
        wTx = w @ x_aug[i]
        delta_w = learning_rate * (y[i] - wTx) * x_aug[i]
        w = w + delta_w

print("Learned weights:", w)

# Compute predicted y values for regression line
y_pred = [w @ x_aug[i] for i in range(x.shape[0])]

# PLOTTING
plt.figure(figsize=(8,5))

# Scatter plot of actual points
plt.scatter(x, y, color='blue', label='Data points', s=50)

# Regression line
plt.plot(x, y_pred, color='red', linewidth=2, label='Regression line')

# Labels and title
plt.xlabel('Number of Claims', fontsize=12)
plt.ylabel('total payment for all the claims in thousands of Swedish Kronor for geographical zones in Sweden', fontsize=7)
plt.title('Linear Regression Fit', fontsize=14)
plt.legend()
plt.grid(True)

# Show plot
plt.show()