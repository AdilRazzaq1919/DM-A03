import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_recall_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
import random

# Step 1: Load and preprocess the dataset
data = pd.read_csv("creditcard.csv")

# Check for missing values in the 'Class' column
print("Missing values in 'Class' column:", data['Class'].isnull().sum())

# Drop rows where 'Class' is NaN
data = data.dropna(subset=['Class'])

# Re-check to confirm no more NaNs in 'Class'
print("Missing values in 'Class' column after dropping rows:", data['Class'].isnull().sum())

# Standardize numerical features
scaler = StandardScaler()
data['Time'] = scaler.fit_transform(data[['Time']])
data['Amount'] = scaler.fit_transform(data[['Amount']])

# Split into features and labels
X = data.drop('Class', axis=1)
y = data['Class']

# Split into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Convert to PyTorch tensors
train_dataset = TensorDataset(torch.tensor(X_train.values, dtype=torch.float32), torch.tensor(y_train.values, dtype=torch.int64))
val_dataset = TensorDataset(torch.tensor(X_val.values, dtype=torch.float32), torch.tensor(y_val.values, dtype=torch.int64))
test_dataset = TensorDataset(torch.tensor(X_test.values, dtype=torch.float32), torch.tensor(y_test.values, dtype=torch.int64))

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

# Step 2: Data Augmentation using geometric distribution masks
def augment_data(tensor, p=0.1):
    augmented_tensor = tensor.clone()
    for i in range(tensor.size(0)):
        for j in range(tensor.size(1)):
            if random.random() < p:
                augmented_tensor[i][j] = tensor[i][j] + random.gauss(0, 0.1)  # Gaussian noise
    return augmented_tensor

augmented_train_dataset = TensorDataset(augment_data(torch.tensor(X_train.values, dtype=torch.float32)), torch.tensor(y_train.values, dtype=torch.int64))
augmented_train_loader = DataLoader(augmented_train_dataset, batch_size=256, shuffle=True)

# Step 3: Transformer-Based Autoencoder
class TransformerAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(TransformerAutoencoder, self).__init__()
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4), num_layers=num_layers)
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=4), num_layers=num_layers)
        self.hidden_dim = hidden_dim
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.reconstruction = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        x = self.embedding(x)
        encoded = self.encoder(x.unsqueeze(1))  # Add a sequence dimension
        decoded = self.decoder(encoded, encoded)  # Using encoded as both target and memory
        reconstructed = self.reconstruction(decoded.squeeze(1))  # Remove the sequence dimension
        return reconstructed

input_dim = X_train.shape[1]
hidden_dim = 64
num_layers = 2

model = TransformerAutoencoder(input_dim, hidden_dim, num_layers)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 4: Contrastive Learning
# Define a contrastive loss function to promote similarity among normal data
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        distance = torch.nn.functional.pairwise_distance(output1, output2)
        loss = (1 - label) * torch.pow(distance, 2) + label * torch.pow(torch.clamp(self.margin - distance, 0, float('inf')), 2)
        return loss.mean()

contrastive_loss = ContrastiveLoss(margin=1.0)

# Step 5: Generative Adversarial Network (GAN)
# A simple GAN with a generator and discriminator for data augmentation
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, z):
        return self.model(z)

# Define the Discriminator class with correct syntax
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()  # Fix: Add the missing opening parenthesis
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Sigmoid output to get a probability between 0 and 1
        )

    def forward(self, x):
        return self.model(x)  # This forward method feeds the data through the model

# GAN Initialization
gan_input_dim = 100  # Latent space dimension
gan_output_dim = input_dim

generator = Generator(gan_input_dim, gan_output_dim)
discriminator = Discriminator(gan_output_dim)

gan_criterion = nn.BCELoss()  # Binary Cross-Entropy Loss for GAN
gan_optimizer_g = optim.Adam(generator.parameters(), lr=0.0002)
gan_optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002)

# Step 6: GAN Training
num_gan_epochs = 20
gan_batch_size = 256

# Training the GAN with attention to batch size consistency
for epoch in range(num_gan_epochs):
    for real_data, _ in train_loader:
        # Get the actual batch size (this helps avoid size mismatch for the last batch)
        current_batch_size = real_data.size(0)

        # Train Discriminator
        noise = torch.randn(current_batch_size, gan_input_dim)  # Generate noise with the correct batch size
        fake_data = generator(noise)

        # Discriminator loss with real and fake data
        real_target = torch.ones(current_batch_size, 1)  # Create real targets with the correct batch size
        fake_target = torch.zeros(current_batch_size, 1)  # Create fake targets with the correct batch size

        real_loss = gan_criterion(discriminator(real_data), real_target)  # Ensure input and target have the same size
        fake_loss = gan_criterion(discriminator(fake_data.detach()), fake_target)

        discriminator_loss = (real_loss + fake_loss) / 2

        gan_optimizer_d.zero_grad()
        discriminator_loss.backward()
        gan_optimizer_d.step()

        # Train Generator
        generator_loss = gan_criterion(discriminator(fake_data), real_target)  # Ensure input and target sizes match

        gan_optimizer_g.zero_grad()
        generator_loss.backward()
        gan_optimizer_g.step()

    print(f"GAN Epoch {epoch + 1}/{num_gan_epochs}, Discriminator Loss: {discriminator_loss.item():.4f}, Generator Loss: {generator_loss.item():.4f}")


# Step 7: Data Augmentation using GAN-generated data
# Generate additional synthetic data using the trained GAN
num_synthetic_samples = 1000
noise = torch.randn(num_synthetic_samples, gan_input_dim)
synthetic_data = generator(noise).detach().numpy()

# Augment the training set with synthetic data
augmented_train_dataset = TensorDataset(torch.tensor(np.vstack((X_train.values, synthetic_data)), dtype=torch.float32), torch.tensor(np.concatenate((y_train.values, np.zeros(num_synthetic_samples))), dtype=torch.int64))
augmented_train_loader = DataLoader(augmented_train_dataset, batch_size=256, shuffle=True)

# Step 8: Train the Transformer-Based Autoencoder with the augmented data
num_epochs = 20
for epoch in range(num_epochs):
    # Training loop with augmented data
    model.train()
    for data, _ in augmented_train_loader:
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, data)
        loss.backward()
        optimizer.step()

    # Validation loop
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data, _ in val_loader:
            outputs = model(data)
            loss = criterion(outputs, data)
            val_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {loss.item():.4f}, Validation Loss: {val_loss / len(val_loader):.4f}")

# Step 9: Anomaly Detection and Evaluation
# Define a threshold for anomaly detection
reconstruction_errors = []
labels = []

# Corrected loop to append each label and reconstruction error
with torch.no_grad():
    for data, label in test_loader:
        outputs = model(data)
        loss = criterion(outputs, data)

        # Add each reconstruction error to the list
        reconstruction_errors.append(loss.item())

        # Corrected method to append individual labels
        labels.extend(label.tolist())  # Changed from append() to extend() for batch labels

threshold = np.percentile(reconstruction_errors, 95)  # Set threshold at 95th percentile

# Determine which instances are anomalies based on the threshold
anomalies = [1 if err > threshold else 0 for err in reconstruction_errors]

# Calculate the precision-recall curve and AUPRC for evaluation
precision, recall, _ = precision_recall_curve(labels, reconstruction_errors)
auprc = auc(recall, precision)
print(f"AUPRC: {auprc:.4f}")

plt.figure()
plt.plot(recall, precision, label="Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve for Anomaly Detection")
plt.show()
