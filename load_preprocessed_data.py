#!/usr/bin/env python3
"""
Script to load and inspect preprocessed TESS data.
"""

import numpy as np
from preprocess_data import load_preprocessed_data

def main():
    print("Loading preprocessed TESS data...")
    print("=" * 50)
    
    # Load the preprocessed data
    X, y = load_preprocessed_data("preprocessed_data")
    
    if X is not None and y is not None:
        print(f"✅ Successfully loaded preprocessed data!")
        print(f"   - Features shape: {X.shape}")
        print(f"   - Labels shape: {y.shape}")
        print(f"   - Data type: {X.dtype}")
        print(f"   - Memory usage: {X.nbytes / 1024 / 1024:.2f} MB")
        
        print(f"\nData statistics:")
        print(f"   - Mean flux: {np.mean(X):.6f}")
        print(f"   - Std flux: {np.std(X):.6f}")
        print(f"   - Min flux: {np.min(X):.6f}")
        print(f"   - Max flux: {np.max(X):.6f}")
        
        print(f"\nLabel distribution:")
        unique_labels, counts = np.unique(y, return_counts=True)
        for label, count in zip(unique_labels, counts):
            print(f"   - Label {label}: {count} samples ({count/len(y)*100:.1f}%)")
        
        print(f"\nSample light curve (first 50 points):")
        print(f"   {X[0, :50]}")
        
    else:
        print("❌ Failed to load preprocessed data!")
        print("Make sure to run preprocess_data.py first.")

if __name__ == "__main__":
    main()
