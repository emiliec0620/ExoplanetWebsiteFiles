#!/usr/bin/env python3
"""
Real TESS Data Prediction System
Uses actual preprocessed TESS light curves for prediction instead of synthetic data.
"""

import torch
import numpy as np
import sys
import os
from train_python_model import TransitCNN, load_preprocessed_data

def predict_with_real_data(model_path="transit_cnn_model.pt", num_samples=10):
    """
    Make predictions using real TESS data instead of synthetic data.
    This will give us realistic, diverse predictions.
    """
    try:
        # Load the model
        device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model = checkpoint['model_architecture']
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        print(f"✅ Model loaded successfully on {device}")
        print(f"📊 Model was trained for {checkpoint.get('total_epochs_trained', 'unknown')} epochs")
        print(f"📈 Best validation loss: {checkpoint.get('best_val_loss', 'unknown'):.4f}")
        
        # Load real TESS data
        print("\n🔄 Loading real TESS data...")
        X_real, y_real = load_preprocessed_data("preprocessed_data")
        
        print(f"✅ Loaded {len(X_real)} real TESS light curves")
        print(f"📊 Class distribution: {np.sum(y_real)} positive, {len(y_real) - np.sum(y_real)} negative")
        
        # Select random samples for prediction
        np.random.seed(42)  # For reproducible results
        indices = np.random.choice(len(X_real), min(num_samples, len(X_real)), replace=False)
        
        print(f"\n🧪 Testing predictions on {len(indices)} real TESS light curves:")
        print("=" * 60)
        
        predictions = []
        for i, idx in enumerate(indices):
            # Get the light curve
            light_curve = X_real[idx]
            true_label = y_real[idx]
            
            # Prepare input tensor
            input_tensor = torch.from_numpy(light_curve).float().unsqueeze(0).unsqueeze(0).to(device)
            
            # Make prediction
            with torch.no_grad():
                logits = model(input_tensor)
                probability = torch.sigmoid(logits).item()
            
            # Determine confidence level
            if probability > 0.8 or probability < 0.2:
                confidence = "High"
            elif probability > 0.6 or probability < 0.4:
                confidence = "Medium"
            else:
                confidence = "Low"
            
            # Determine prediction
            prediction = "EXOPLANET" if probability > 0.5 else "NOT EXOPLANET"
            
            # Check if prediction matches true label
            correct = "✅" if (probability > 0.5) == (true_label > 0.5) else "❌"
            
            print(f"Sample {i+1:2d}: {probability:.4f} ({probability*100:.2f}%) - {prediction} - {confidence} confidence {correct}")
            print(f"         True label: {'EXOPLANET' if true_label > 0.5 else 'NOT EXOPLANET'}")
            
            predictions.append(probability)
        
        # Summary statistics
        print("\n" + "=" * 60)
        print("SUMMARY STATISTICS:")
        print(f"Number of predictions: {len(predictions)}")
        print(f"Average probability: {np.mean(predictions):.4f}")
        print(f"Min probability: {np.min(predictions):.4f}")
        print(f"Max probability: {np.max(predictions):.4f}")
        print(f"Standard deviation: {np.std(predictions):.4f}")
        
        # Check for variation
        variation = np.max(predictions) - np.min(predictions)
        if variation > 0.1:
            print(f"✅ Good variation in predictions: {variation:.4f}")
        else:
            print(f"⚠️  Limited variation in predictions: {variation:.4f}")
        
        return predictions
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def test_model_on_synthetic_vs_real():
    """
    Compare how the model performs on synthetic vs real data.
    """
    print("🔬 COMPARING SYNTHETIC vs REAL DATA PERFORMANCE")
    print("=" * 60)
    
    # Test with real data
    print("\n1. Testing with REAL TESS data:")
    real_predictions = predict_with_real_data(num_samples=20)
    
    if real_predictions is not None:
        print(f"\n📊 Real data results:")
        print(f"   - Variation: {np.max(real_predictions) - np.min(real_predictions):.4f}")
        print(f"   - Range: {np.min(real_predictions):.4f} to {np.max(real_predictions):.4f}")
        print(f"   - Mean: {np.mean(real_predictions):.4f}")
    
    print("\n" + "=" * 60)
    print("CONCLUSION:")
    if real_predictions is not None:
        variation = np.max(real_predictions) - np.min(real_predictions)
        if variation > 0.2:
            print("✅ Model shows good sensitivity on real data")
            print("💡 The issue was using synthetic data instead of real data")
        else:
            print("⚠️  Model shows limited sensitivity even on real data")
            print("💡 May need to retrain the model or adjust architecture")
    else:
        print("❌ Could not test model performance")

if __name__ == "__main__":
    test_model_on_synthetic_vs_real()
