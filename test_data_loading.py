#!/usr/bin/env python3
"""
Test script to verify data loading works correctly.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
from preprocess_data import load_preprocessed_data

# For reproducibility
torch.manual_seed(42)
np.random.seed(42)

print("Testing data loading...")
print("=" * 50)

# Load the preprocessed TESS data
print("Loading preprocessed TESS data...")
X_real, y_real = load_preprocessed_data("preprocessed_data")

if X_real is None or y_real is None:
    print("❌ Failed to load preprocessed data!")
    print("Please run preprocess_data.py first to generate the preprocessed data.")
    exit(1)

print(f"✅ Loaded preprocessed data: {X_real.shape[0]} samples")
print(f"   - Data shape: {X_real.shape}")
print(f"   - Data type: {X_real.dtype}")
print(f"   - Label distribution: {np.unique(y_real, return_counts=True)}")

# Use a subset for testing
print("\nUsing subset for testing...")
subset_size = 1000
X_subset = X_real[:subset_size]
y_subset = y_real[:subset_size]

# Generate some positive examples
print("Generating synthetic transit examples...")
num_positive = 500
X_positive = np.zeros((num_positive, 2048))

for i in range(num_positive):
    base_idx = i % len(X_subset)
    base_curve = X_subset[base_idx].copy()
    
    # Add a synthetic transit
    transit_start = np.random.randint(200, 2048 - 200)
    transit_duration = np.random.randint(20, 80)
    transit_depth = np.random.uniform(0.005, 0.02)
    
    for j in range(transit_duration):
        if transit_start + j < 2048:
            base_curve[transit_start + j] -= transit_depth
    
    X_positive[i] = base_curve

y_positive = np.ones(num_positive)

# Combine data
X = np.vstack([X_subset, X_positive])
y = np.hstack([y_subset, y_positive])

print(f"Combined dataset: {len(X)} samples")
print(f"  - Negative: {len(X_subset)}")
print(f"  - Positive: {len(X_positive)}")

# Test dataset creation
class LightCurveDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.from_numpy(features).float().unsqueeze(1)
        self.labels = torch.from_numpy(labels).float().unsqueeze(1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Create dataset
dataset = LightCurveDataset(X, y)
print(f"✅ Dataset created successfully: {len(dataset)} samples")

# Test data loader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
print(f"✅ DataLoader created successfully")

# Test one batch
for batch_idx, (inputs, labels) in enumerate(dataloader):
    print(f"✅ Batch {batch_idx + 1}: inputs shape {inputs.shape}, labels shape {labels.shape}")
    if batch_idx >= 2:  # Test first 3 batches
        break

print("\n🎉 Data loading test completed successfully!")
print("The training script should work with the preprocessed data.")
