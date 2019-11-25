import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (20., 10.)

dir_path = os.path.dirname(os.path.realpath(__file__))

# Reading Data
data = pd.read_csv(os.path.join(dir_path, 'headbrain.csv'))
print(data.shape)

print(data.head())

# Collecting X and Y
X = data['Head Size(cm^3)'].values
Y = data['Brain Weight(grams)'].values

# Mean X and Y
mean_X = np.mean(X)
mean_Y = np.mean(Y)

# Total number of values
cnt = len(X)

# Using the formula to calculate b1 and b2
number = 0
denom = 0
for i in range(cnt):
	number += (X[i] - mean_X) * (Y[i] - mean_Y)
	denom += (X[i] - mean_X) ** 2
b1 = number / denom
b0 = mean_Y - (b1 * mean_X)

print(b1, b0)

# Plotting Values and Regression Line
max_x = np.max(X) + 100
min_x = np.min(X) - 100

# Calculation Line values x and y
x = np.linspace(min_x, max_x, 1000)
y = b0 + b1 * x

# Plating Line
plt.plot(x, y, color='#58d970', label='Regression Line')
# Plating Scatter Plot
plt.scatter(X, Y, c='#ef5424', label='Scatter Plot')

# Visualise
plt.xlabel('Head size in cm3')
plt.ylabel('Brain Weight in grams')
plt.legend()
plt.show()

ss_t = 0
ss_r = 0
for i in range(cnt):
	y_pred = b0 + b1 * X[i]
	ss_t += (Y[i] - mean_Y) ** 2
	ss_r += (Y[i] - y_pred) ** 2
r2 = 1 - (ss_r/ss_t)
print(r2)
