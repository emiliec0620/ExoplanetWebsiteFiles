#!/usr/bin/env python3
"""
1D CNN-based exoplanet detection model for light curve time-series data.
This captures local temporal patterns and features in light curves.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
from preprocess_data import load_preprocessed_data

# For reproducibility
torch.manual_seed(42)
np.random.seed(42)

class LightCurveCNN1D(nn.Module):
    """1D CNN model for light curve classification."""
    def __init__(self, input_size=2048, num_classes=1):
        super(LightCurveCNN1D, self).__init__()
        
        # 1D Convolutional layers for temporal pattern recognition
        self.conv_layers = nn.Sequential(
            # First conv block
            nn.Conv1d(1, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.1),
            
            # Second conv block
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),
            
            # Third conv block
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3),
            
            # Fourth conv block
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.4),
            
            # Fifth conv block
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # Global average pooling
            nn.Dropout(0.5)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights properly."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.uniform_(m.bias, -0.1, 0.1)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Reshape input: (batch_size, features) -> (batch_size, 1, features)
        x = x.unsqueeze(1)  # Add channel dimension
        
        # Convolutional layers
        x = self.conv_layers(x)
        
        # Flatten for classification
        x = x.view(x.size(0), -1)
        
        # Classification
        output = self.classifier(x)
        return output.squeeze(-1)

class LightCurveDataset(Dataset):
    """Custom PyTorch Dataset for light curve data."""
    def __init__(self, features, labels, augment=False):
        self.features = torch.from_numpy(features).float()
        self.labels = torch.from_numpy(labels).float()
        self.augment = augment

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx].clone()
        
        if self.augment and torch.rand(1) < 0.5:
            # Add small amount of noise
            noise = torch.randn_like(feature) * 0.01
            feature = feature + noise
            
            # Random scaling
            scale = torch.rand(1) * 0.1 + 0.95  # 0.95 to 1.05
            feature = feature * scale
            
            # Random shifting
            shift = torch.randn(1) * 0.02
            feature = feature + shift
        
        return feature, self.labels[idx]

if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading preprocessed TESS data...")
    X_real, y_real = load_preprocessed_data("preprocessed_data")
    print(f"✅ Loaded preprocessed data: {len(X_real)} samples")
    
    # Create dataset
    dataset = LightCurveDataset(X_real, y_real, augment=True)
    
    # Split data
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    print(f"Data prepared: {len(train_dataset)} training samples, {len(val_dataset)} validation samples.")
    
    # Create model
    model = LightCurveCNN1D(input_size=2048, num_classes=1)
    model.to(device)
    
    print(f"Model Architecture:\n{model}")
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Training loop
    num_epochs = 30
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    train_losses = []
    val_losses = []
    
    print("\nStarting 1D CNN model training...")
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            print(f"Training batch {batch_idx+1}/{len(train_loader)}")
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1}/{num_epochs} -> Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
            }, 'cnn1d_exoplanet_model.pt')
            print(f"⭐ NEW BEST! Saving model...")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"🛑 Early stopping triggered after {patience} epochs without improvement")
                break
    
    print(f"\n🎉 Training finished!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    # Load best model
    checkpoint = torch.load('cnn1d_exoplanet_model.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Test sensitivity
    print("\n🧪 Testing 1D CNN model sensitivity...")
    model.eval()
    test_predictions = []
    
    with torch.no_grad():
        for i in range(min(50, len(val_dataset))):
            data, _ = val_dataset[i]
            data = data.unsqueeze(0).to(device)
            output = model(data)
            prob = torch.sigmoid(output).item()
            test_predictions.append(prob)
    
    if test_predictions:
        min_prob = min(test_predictions)
        max_prob = max(test_predictions)
        avg_prob = np.mean(test_predictions)
        variation = max_prob - min_prob
        
        print(f"📊 1D CNN Model Sensitivity Test:")
        print(f"   Range: {min_prob:.4f} - {max_prob:.4f}")
        print(f"   Average: {avg_prob:.4f}")
        print(f"   Variation: {variation:.4f}")
        
        if variation > 0.1:
            print(f"   ✅ EXCELLENT sensitivity: {variation:.4f}")
        elif variation > 0.05:
            print(f"   ✅ GOOD sensitivity: {variation:.4f}")
        elif variation > 0.01:
            print(f"   ⚠️  MODERATE sensitivity: {variation:.4f}")
        else:
            print(f"   ❌ POOR sensitivity: {variation:.4f}")
    
    print(f"\n💾 1D CNN model saved to 'cnn1d_exoplanet_model.pt'")
