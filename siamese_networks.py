import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

'''
Key Characteristics
Two Branches with Shared Weights: Siamese networks have two parallel branches, each processing a separate input. The critical aspect is that these branches share the same weights, ensuring they generate consistent embeddings regardless of which branch the input goes through.
Input Pairs: Given two inputs, a Siamese network processes each through its respective branch to generate their embeddings. The embeddings are then compared to determine how similar or dissimilar the inputs are.
Distance Metric: A distance metric (like Euclidean distance or cosine similarity) is used to quantify the similarity between embeddings. This metric can be part of the loss function during training, guiding the network to learn embeddings that reflect the similarity or difference between input pairs'''


# Define the Siamese Network
class SiameseNetwork(nn.Module):

    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.conn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )

    def forward(self, x):
        x = self.conn(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# Contrastive loss for Siamese networks
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = torch.nn.functional.pairwise_distance(output1, output2)
        loss = (1 - label) * torch.pow(euclidean_distance, 2) + label * torch.pow(
            torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        return loss


# Custom dataset for Siamese network with MNIST
class SiameseMNISTDataset(Dataset):
    def __init__(self, mnist_data):
        self.mnist_data = mnist_data

    def __len__(self):
        return len(self.mnist_data)

    def __getitem__(self, idx):
        image, label = self.mnist_data[idx]
        # Randomly select another image to form a pair
        other_idx = np.random.randint(0, len(self.mnist_data))
        other_image, other_label = self.mnist_data[other_idx]
        same_class = 1 if label == other_label else 0  # 1 if same class, 0 otherwise
        return (image, other_image), same_class


# Load the MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),  # Normalization for MNIST
])

mnist_train = torchvision.datasets.MNIST("data/", train=True, download=True, transform=transform)
mnist_test = torchvision.datasets.MNIST("data/", train=False, download=True, transform=transform)

# Create training and testing datasets
train_dataset = SiameseMNISTDataset(mnist_train)
test_dataset = SiameseMNISTDataset(mnist_test)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)

# Initialize the Siamese network, optimizer, and loss function
model = SiameseNetwork()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
contrastive_loss = ContrastiveLoss(margin=1.0)

# Training the Siamese Network
num_epochs = 5
for epoch in range(num_epochs):
    total_loss = 0
    for images, labels in train_loader:
        img1, img2 = images

        # Forward pass
        output1 = model(img1)
        output2 = model(img2)
        loss = contrastive_loss(output1, output2, labels)
        loss = torch.mean(loss)

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"Epoch {epoch + 1}, Total Loss: {total_loss:.4f}")

# Create a support set with one example per class (one-shot context)
support_set = []
classes = set()

for data in mnist_train:
    image, label = data
    if label not in classes:
        support_set.append((image, label))
        classes.add(label)
    if len(support_set) == 10:
        break  # We have one example per class


# Function to classify a query image using the support set and trained Siamese network
def classify_query_image(query_image, support_set, model):
    query_embedding = model(query_image.unsqueeze(0))  # Generate embedding for query image
    min_distance = float('inf')
    predicted_class = None

    # Compare the query embedding with each embedding in the support set
    for (support_image, label) in support_set:
        support_embedding = model(support_image.unsqueeze(0))
        distance = torch.nn.functional.pairwise_distance(query_embedding, support_embedding)
        if distance < min_distance:
            min_distance = distance
            predicted_class = label

    return predicted_class


# Inference with the Siamese Network
# Inference with the Siamese Network
query_images = [mnist_test[i][0] for i in range(10)]
predicted_classes = []

for query_image in query_images:
    predicted_class = classify_query_image(query_image, support_set, model)
    predicted_classes.append(predicted_class)

# Display the query images and their predicted classes
plt.figure(figsize=(10, 2))
for i, (img, predicted_class) in enumerate(zip(query_images, predicted_classes)):
    plt.subplot(1, 10, i + 1)
    plt.imshow(img.squeeze(), cmap='gray')
    plt.title(f"Pred: {predicted_class}")
    plt.axis('off')
plt.show()
