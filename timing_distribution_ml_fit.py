mport torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

#data
data_cell_time = np.load('data/rev_time.npy',allow_pickle=True)
data_cell_time = [(value, count) for value, count in data_cell_time if -12.5 <= value <= 12.5]
filtered_table = [(x, y) for x, y in data_cell_time if y > 1]
filtered_table = np.array(filtered_table)
data_cell_time = filtered_table


X = data_cell_time[:,0:1]
y = data_cell_time[:,1:2]

# add noise to reduce overfitting
y += 0.2 * (np.random.randn(1))

# Convert data to PyTorch tensors
X_tensor = torch.FloatTensor(X)
y_tensor = torch.FloatTensor(y)

# Define the neural network model
class RegressionModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RegressionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.act1 = nn.Tanh()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.act2 = nn.Tanh()
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.fc3(x)
        return x

# Instantiate the model
input_size = 1  # One input feature (X)
hidden_size = 10  # Number of neurons in the hidden layer
output_size = 1  # One output (predicted y)
model = RegressionModel(input_size, hidden_size, output_size)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# Training loop
num_epochs = 30000
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor.view(-1, 1))

    # Backpropagation and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')

# Convert the model's predictions to numpy
predicted = model(X_tensor).detach().numpy()

# Plot the original data and the regression line
plt.scatter(X, y, label='Original Data',s=0.1)
plt.scatter(X, predicted, label='Regression Line', color='red',s=1)
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()

import numpy as np

# Sample scatter plot data (x and y values)
x_values = data_cell_time[:,0:1]
y_values = predicted  # Weighted values

# Calculate the weighted sum of x-values
weighted_sum_x = np.sum(x_values * y_values)

# Calculate the total weight (sum of all y-values)
total_weight = np.sum(y_values)

# Compute the weighted average
weighted_average_pred = weighted_sum_x / total_weight


x_values = data_cell_time[:,0:1]
y_values = data_cell_time[:,1:2]  # Weighted values

# Calculate the weighted sum of x-values
weighted_sum_x = np.sum(x_values * y_values)

# Calculate the total weight (sum of all y-values)
total_weight = np.sum(y_values)

# Compute the weighted average
weighted_average_true = weighted_sum_x / total_weight


print(f"Weighted Average of predicted X: {weighted_average_pred}")
print(f"Weighted Average of true X: {weighted_average_true}")

# Sample scatter plot data (x and y values)
x_values = data_cell_time[:,0:1]
y_values = predicted  # Weighted values

# Calculate the weighted mean of x-values
weighted_mean_x = np.sum(x_values * y_values) / np.sum(y_values)

# Calculate the sum of weighted squared differences
weighted_squared_diff_sum = np.sum(y_values * (x_values - weighted_mean_x)**2)

# Divide by the total weight
total_weight = np.sum(y_values)
weighted_std_pred_x = np.sqrt(weighted_squared_diff_sum / total_weight)

x_values = data_cell_time[:,0:1]
y_values = data_cell_time[:,1:2]  # Weighted valu

weighted_mean_x = np.sum(x_values * y_values) / np.sum(y_values)

# Calculate the sum of weighted squared differences
weighted_squared_diff_sum = np.sum(y_values * (x_values - weighted_mean_x)**2)

# Divide by the total weight
total_weight = np.sum(y_values)
weighted_std_x = np.sqrt(weighted_squared_diff_sum / total_weight)


print(f"Weighted Standard Deviation of predicted X: {weighted_std_pred_x}")
print(f"Weighted Standard Deviation true X: {weighted_std_x}")



