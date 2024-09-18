import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(Net, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_size))
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = torch.sigmoid(x)
        return self.output_layer(x)


vanishing_gradients = []
exploding_gradients = []
input_size = 1
hidden_size = 10
num_layers = 10
model = Net(input_size, hidden_size, num_layers)

# Input and target
x = torch.ones((1, input_size))
target = torch.tensor([[1.0]])  # Reshape to (1, 1)

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

output = model(x)
loss = criterion(output, target)
loss.backward()

# Display the gradients for each layer
print("Gradients for each layer:")
for i, layer in enumerate(model.layers):
    print(f"Layer {i + 1} gradient norm: {layer.weight.grad.norm().item()}")
    vanishing_gradients.append(layer.weight.grad.norm().item())

# Optional: Observe exploding gradients by initializing weights with large values
for layer in model.layers:
    nn.init.normal_(layer.weight, mean=0, std=10)  # Large weights for exploding gradients

output = model(x)
loss = criterion(output, target)
optimizer.zero_grad()
loss.backward()

print("\nGradients after large weight initialization (exploding gradients):")
for i, layer in enumerate(model.layers):
    print(f"Layer {i + 1} gradient norm: {layer.weight.grad.norm().item()}")
    exploding_gradients.append(layer.weight.grad.norm().item())

# Plotting
plt.figure(figsize=(10, 5))
layers = list(range(1, num_layers + 1))

# Plot vanishing gradients
plt.plot(layers, vanishing_gradients, marker='o', label='Vanishing Gradients', color='blue')

# Plot exploding gradients
plt.plot(layers, exploding_gradients, marker='x', label='Exploding Gradients', color='red')

# Labels and title
plt.xlabel('Layer')
plt.ylabel('Gradient Norm')
plt.title('Vanishing and Exploding Gradients in a Deep Neural Network')
plt.yscale('log')  # Use logarithmic scale for better visualization
plt.legend()
plt.grid(True)

# Display the plot
plt.show()
