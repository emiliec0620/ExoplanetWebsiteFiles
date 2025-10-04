import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import matplotlib.pyplot as plt
from preprocess_data import load_preprocessed_data

# For reproducibility
torch.manual_seed(42)
np.random.seed(42)

## ----------------------------------
## 1. DATA LOADING & PREPARATION
## ----------------------------------


class LightCurveDataset(Dataset):
    """Custom PyTorch Dataset for light curve data."""
    def __init__(self, features, labels):
        # Add a channel dimension for the CNN (N, C, L)
        # Ensure both features and labels are float32 to match model parameters
        self.features = torch.from_numpy(features).float().unsqueeze(1)
        self.labels = torch.from_numpy(labels).float().unsqueeze(1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

if __name__ == "__main__":
    # Load the preprocessed TESS data
    print("Loading preprocessed TESS data...")
    X_real, y_real = load_preprocessed_data("preprocessed_data")

    if X_real is None or y_real is None:
        print("❌ Failed to load preprocessed data!")
        print("Please run preprocess_data.py first to generate the preprocessed data.")
        exit(1)

    print(f"✅ Loaded preprocessed data: {X_real.shape[0]} samples")

    # Use the real labeled data from TFOPWG dispositions
    X = X_real
    y = y_real

    print(f"Using real labeled data: {len(X)} total samples")
    print(f"  - Data shape: {X.shape}")

    # Print class distribution
    unique_labels, counts = np.unique(y, return_counts=True)
    print(f"Class distribution:")
    for label, count in zip(unique_labels, counts):
        class_name = {0: 'Negative (FP/Unknown)', 1: 'Positive (CP/KP/PC)'}[int(label)]
        percentage = count / len(y) * 100
        print(f"  - {class_name}: {count} samples ({percentage:.1f}%)")

    # If we have very imbalanced data, we might want to balance it
    if len(unique_labels) == 2:
        pos_count = counts[1] if unique_labels[1] == 1 else counts[0]
        neg_count = counts[0] if unique_labels[1] == 1 else counts[1]
        
        if pos_count > 0 and neg_count > 0:
            # Calculate the ratio
            ratio = neg_count / pos_count
            print(f"Class ratio (neg/pos): {ratio:.2f}")
            
            if ratio > 10:  # Very imbalanced
                print("⚠️ Highly imbalanced dataset detected. Consider using class weights or data augmentation.")
            elif ratio < 0.1:  # Very imbalanced the other way
                print("⚠️ Highly imbalanced dataset detected. Consider using class weights or data augmentation.")
            else:
                print("✅ Dataset appears reasonably balanced.")

    # Shuffle the dataset
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]

    # Create dataset and split into training and validation sets
    full_dataset = LightCurveDataset(X, y)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Create DataLoaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    print(f"Data prepared: {len(train_dataset)} training samples, {len(val_dataset)} validation samples.")

    # Print class distribution
    train_labels = [y[i] for i in train_dataset.indices]
    val_labels = [y[i] for i in val_dataset.indices]

    train_pos = sum(train_labels)
    train_neg = len(train_labels) - train_pos
    val_pos = sum(val_labels)
    val_neg = len(val_labels) - val_pos

    print(f"Training set - Positive: {train_pos}, Negative: {train_neg}")
    print(f"Validation set - Positive: {val_pos}, Negative: {val_neg}")


## -------------------------
## 2. MODEL ARCHITECTURE (CNN)
## -------------------------
class TransitCNN(nn.Module):
    """A simple 1D CNN for detecting exoplanet transits."""
    def __init__(self):
        super(TransitCNN, self).__init__()
        self.network = nn.Sequential(
            # Input shape: (batch_size, 1, 2048)
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),
            # Shape: (batch_size, 16, 512)

            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),
            # Shape: (batch_size, 32, 128)

            nn.Flatten(),
            nn.Linear(32 * 128, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid() # Output a probability between 0 and 1
        )

    def forward(self, x):
        return self.network(x)

if __name__ == "__main__":
    # Set device and ensure model uses float32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransitCNN().to(device).float()
    print("Model Architecture:\n", model)
    print(f"Using device: {device}")


    ## -------------------------
    ## 3. TRAINING THE MODEL
    ## -------------------------
    # Loss function and optimizer
    criterion = nn.BCELoss() # Binary Cross-Entropy for binary classification
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 10

    # Lists to store loss history for plotting
    train_loss_history = []
    val_loss_history = []

    print("\nStarting model training...")

    for epoch in range(num_epochs):
        # --- Training Phase ---
        model.train()
        running_train_loss = 0.0
        for inputs, labels in train_loader:
            # Move data to the same device as the model
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * inputs.size(0)

        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        train_loss_history.append(epoch_train_loss)

        # --- Validation Phase ---
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                # Move data to the same device as the model
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * inputs.size(0)

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_loss_history.append(epoch_val_loss)

        print(f"Epoch {epoch+1}/{num_epochs} -> Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")

    print("Training finished! 🎉")

    # Save the trained model
    model_save_path = "transit_cnn_model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_architecture': model,
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': num_epochs,
        'train_loss_history': train_loss_history,
        'val_loss_history': val_loss_history,
        'device': device
    }, model_save_path)
    print(f"Model saved to {model_save_path}")


    ## -------------------------
    ## 4. VISUALIZING RESULTS
    ## -------------------------
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_history, label='Training Loss')
    plt.plot(val_loss_history, label='Validation Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    # You can now use model.eval() and pass new light curves to it for predictions.
    # For example:
    # test_lc = simulate_light_curve(has_transit=True)
    # test_tensor = torch.from_numpy(test_lc).float().unsqueeze(0).unsqueeze(0).to(device) # Shape: (1, 1, 2048)
    # prediction = model(test_tensor)
    # print(f"Prediction for a transit light curve: {prediction.item():.4f}")

# Function to load a saved model
def load_model(model_path, device):
    """Load a saved model from file."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = checkpoint['model_architecture']
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model, checkpoint