#!/usr/bin/env python3
"""
Transformer-based exoplanet detection model for light curve time-series data.
This uses self-attention mechanisms to capture complex temporal patterns.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import math
from preprocess_data import load_preprocessed_data

# For reproducibility
torch.manual_seed(42)
np.random.seed(42)

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class LightCurveTransformer(nn.Module):
    """Transformer-based model for light curve classification."""
    def __init__(self, input_size=128, d_model=256, nhead=8, num_layers=4, dropout=0.1):
        super(LightCurveTransformer, self).__init__()
        
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(1, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len=input_size)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Global attention pooling
        self.global_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(128, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights properly."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.uniform_(m.bias, -0.1, 0.1)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Reshape and project input
        x = x.view(batch_size, -1, 1)  # (batch, seq_len, 1)
        x = self.input_projection(x)   # (batch, seq_len, d_model)
        
        # Add positional encoding
        x = x.transpose(0, 1)  # (seq_len, batch, d_model)
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # (batch, seq_len, d_model)
        
        # Transformer encoding
        encoded = self.transformer(x)  # (batch, seq_len, d_model)
        
        # Global attention pooling
        # Use a learnable query for global attention
        query = torch.mean(encoded, dim=1, keepdim=True)  # (batch, 1, d_model)
        attn_out, _ = self.global_attention(query, encoded, encoded)
        pooled = attn_out.squeeze(1)  # (batch, d_model)
        
        # Classification
        output = self.classifier(pooled)
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
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)  # Smaller batch for transformer
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    print(f"Data prepared: {len(train_dataset)} training samples, {len(val_dataset)} validation samples.")
    
    # Create model
    model = LightCurveTransformer(input_size=2048, d_model=256, nhead=8, num_layers=4, dropout=0.1)
    model.to(device)
    
    print(f"Model Architecture:\n{model}")
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)  # AdamW for transformers
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-6)
    
    # Training loop
    num_epochs = 25
    best_val_loss = float('inf')
    patience = 8
    patience_counter = 0
    
    train_losses = []
    val_losses = []
    
    print("\nStarting Transformer model training...")
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
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
        
        scheduler.step()
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
            }, 'transformer_exoplanet_model.pt')
            print(f"⭐ NEW BEST! Saving model...")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"🛑 Early stopping triggered after {patience} epochs without improvement")
                break
    
    print(f"\n🎉 Training finished!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    # Load best model
    checkpoint = torch.load('transformer_exoplanet_model.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Test sensitivity
    print("\n🧪 Testing Transformer model sensitivity...")
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
        
        print(f"📊 Transformer Model Sensitivity Test:")
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
    
    print(f"\n💾 Transformer model saved to 'transformer_exoplanet_model.pt'")
