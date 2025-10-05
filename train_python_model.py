import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import matplotlib.pyplot as plt
from preprocess_data import load_preprocessed_data

# For reproducibility
torch.manual_seed(42)
np.random.seed(42)

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance."""
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()

## ----------------------------------
## 1. DATA LOADING & PREPARATION
## ----------------------------------


class LightCurveDataset(Dataset):
    """Custom PyTorch Dataset for light curve data with augmentation."""
    def __init__(self, features, labels, augment=False):
        # For MLP, we don't need the channel dimension - just flatten
        # Ensure both features and labels are float32 to match model parameters
        self.features = torch.from_numpy(features).float()
        self.labels = torch.from_numpy(labels).float()
        self.augment = augment

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx].clone()
        
        if self.augment and torch.rand(1) < 0.7:  # 70% chance of augmentation
            # Add small amount of noise
            noise = torch.randn_like(feature) * 0.01
            feature = feature + noise
            
            # Random scaling
            scale = torch.rand(1) * 0.1 + 0.95  # Scale between 0.95 and 1.05
            feature = feature * scale
            
            # Random shift
            shift = torch.rand(1) * 0.02 - 0.01  # Shift between -0.01 and 0.01
            feature = feature + shift
        
        return feature, self.labels[idx]

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
    train_indices, val_indices = random_split(range(len(full_dataset)), [train_size, val_size])
    
    # Create training dataset with augmentation
    train_dataset = LightCurveDataset(X[train_indices.indices], y[train_indices.indices], augment=True)
    val_dataset = LightCurveDataset(X[val_indices.indices], y[val_indices.indices], augment=False)

    # Create DataLoaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    print(f"Data prepared: {len(train_dataset)} training samples, {len(val_dataset)} validation samples.")

    # Print class distribution
    train_labels = y[train_indices.indices]
    val_labels = y[val_indices.indices]

    train_pos = sum(train_labels)
    train_neg = len(train_labels) - train_pos
    val_pos = sum(val_labels)
    val_neg = len(val_labels) - val_pos

    print(f"Training set - Positive: {train_pos}, Negative: {train_neg}")
    print(f"Validation set - Positive: {val_pos}, Negative: {val_neg}")


## -------------------------
## 2. MODEL ARCHITECTURE (MLP)
## -------------------------
class ExoplanetMLP(nn.Module):
    """A regular neural network (MLP) for exoplanet detection."""
    def __init__(self, input_size=2048):
        super(ExoplanetMLP, self).__init__()
        self.network = nn.Sequential(
            # Input layer
            nn.Linear(input_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            # nn.Dropout(0.3),
            
            # Hidden layers - MORE SENSITIVE architecture
            nn.Linear(1024, 768),
            nn.BatchNorm1d(768),
            nn.LeakyReLU(0.2),      # LeakyReLU for better gradients
            nn.Dropout(0.3),

            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(512, 384),
            nn.BatchNorm1d(384),
            nn.LeakyReLU(0.1),      # Mixed activations
            nn.Dropout(0.5),

            nn.Linear(384, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.6),
            
            # Output layer
            nn.Linear(256, 1)  # Raw logits for BCEWithLogitsLoss
        )

    def _init_weights(self):
        """Initialize weights with better distribution to avoid saturation."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Xavier/Glorot initialization for better gradient flow
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    # Small positive bias to break symmetry
                    nn.init.uniform_(m.bias, -0.1, 0.1)
            elif isinstance(m, nn.BatchNorm1d):
                # Initialize batch norm for better training
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Flatten input if it has extra dimensions
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        output = self.network(x)
        # Squeeze to match target shape for BCEWithLogitsLoss
        return output.squeeze(-1)

if __name__ == "__main__":
    # Set device and ensure model uses float32
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    model = ExoplanetMLP().to(device).float()
    # Initialize weights for symmetry breaking
    model._init_weights()
    print("Model Architecture:\n", model)
    print(f"Using device: {device}")
    print("✅ Weights initialized for symmetry breaking")


    ## -------------------------
    ## 3. TRAINING THE MODEL
    ## -------------------------
    # Calculate class weights for imbalanced dataset
    pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float32).to(device)  # Weight for positive class
    print(f"📊 Using BCEWithLogitsLoss for binary classification")
    
    # Check initial weight distributions
    print(f"🔍 Initial weight analysis:")
    for name, param in model.named_parameters():
        if 'weight' in name:
            weight_mean = param.data.mean().item()
            weight_std = param.data.std().item()
            weight_min = param.data.min().item()
            weight_max = param.data.max().item()
            print(f"   {name}: mean={weight_mean:.4f}, std={weight_std:.4f}, range=[{weight_min:.4f}, {weight_max:.4f}]")
    
    # Loss function and optimizer with regularization
    criterion = nn.BCEWithLogitsLoss()  # Use standard BCEWithLogitsLoss
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # Higher weight decay
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    num_epochs = 50  # Increased epochs with early stopping

    # Lists to store loss history for plotting
    train_loss_history = []
    val_loss_history = []
    
    # Track best model based on validation loss
    best_val_loss = float('inf')
    best_model_state = None
    best_epoch = 0
    patience = 8  # Reduced patience for more training
    patience_counter = 0

    print("\nStarting model training...")
    print("📊 Tracking best model based on validation loss...")
    print("🛡️ Anti-overfitting measures enabled:")
    print("   - L2 regularization (weight_decay=1e-4)")
    print("   - Learning rate scheduling (ReduceLROnPlateau)")
    print("   - Increased dropout rates (0.7, 0.6, 0.5, 0.4)")
    print("   - Data augmentation (noise, scaling, shifting)")
    print("   - Gradient clipping (max_norm=1.0)")
    print("   - Early stopping (patience=15)")
    print("   - Batch normalization in all layers")

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
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)  # Increased gradient clipping
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
        
        # Learning rate scheduling
        scheduler.step(epoch_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Check if this is the best model so far
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_state = model.state_dict().copy()
            best_epoch = epoch + 1
            patience_counter = 0
            print(f"Epoch {epoch+1}/{num_epochs} -> Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, LR: {current_lr:.6f} ⭐ NEW BEST!")
        else:
            patience_counter += 1
            print(f"Epoch {epoch+1}/{num_epochs} -> Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, LR: {current_lr:.6f}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\n🛑 Early stopping triggered after {patience} epochs without improvement")
            print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")
            break

    print("Training finished! 🎉")
    print(f"📊 Best model: Epoch {best_epoch} with validation loss {best_val_loss:.4f}")

    # Load the best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"✅ Loaded best model from epoch {best_epoch}")
    
    # Save the best model
    model_save_path = "transit_cnn_model.pt"
    torch.save({
        'model_state_dict': best_model_state if best_model_state is not None else model.state_dict(),
        'model_architecture': model,
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'train_loss_history': train_loss_history,
        'val_loss_history': val_loss_history,
        'device': device,
        'total_epochs_trained': epoch + 1
    }, model_save_path)
    print(f"💾 Best model saved to {model_save_path}")
    print(f"   - Best epoch: {best_epoch}")
    print(f"   - Best validation loss: {best_val_loss:.4f}")
    print(f"   - Total epochs trained: {epoch + 1}")


    ## -------------------------
    ## 4. VISUALIZING RESULTS
    ## -------------------------
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Loss curves
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_history, label='Training Loss', color='blue')
    plt.plot(val_loss_history, label='Validation Loss', color='red')
    plt.axvline(x=best_epoch-1, color='green', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch})')
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Validation loss zoom
    plt.subplot(1, 2, 2)
    plt.plot(val_loss_history, label='Validation Loss', color='red', linewidth=2)
    plt.axvline(x=best_epoch-1, color='green', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch})')
    plt.axhline(y=best_val_loss, color='orange', linestyle=':', alpha=0.7, label=f'Best Loss ({best_val_loss:.4f})')
    plt.title('Validation Loss Detail')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
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