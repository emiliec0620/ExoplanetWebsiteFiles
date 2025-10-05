#!/usr/bin/env python3
"""
Quick test to check if the improved model shows better sensitivity.
"""

import torch
import numpy as np
import sys
import os
from train_python_model import ExoplanetMLP, load_preprocessed_data

def test_model_sensitivity():
    """Test the improved model's sensitivity."""
    print("🧪 Testing Improved Model Sensitivity")
    print("=" * 50)
    
    try:
        # Load the model
        device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        checkpoint = torch.load('transit_cnn_model.pt', map_location=device, weights_only=False)
        model = checkpoint['model_architecture']
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        print(f"✅ Model loaded on {device}")
        print(f"📊 Best validation loss: {checkpoint.get('best_val_loss', 'unknown'):.4f}")
        print(f"📈 Best epoch: {checkpoint.get('epoch', 'unknown')}")
        
        # Load real TESS data
        X_real, y_real = load_preprocessed_data("preprocessed_data")
        print(f"✅ Loaded {len(X_real)} real TESS light curves")
        
        # Test with different numbers of samples
        for num_samples in [10, 25, 50]:
            print(f"\n🔬 Testing with {num_samples} samples:")
            print("-" * 40)
            
            # Select random samples
            np.random.seed(42)  # For reproducible results
            indices = np.random.choice(len(X_real), min(num_samples, len(X_real)), replace=False)
            
            predictions = []
            for idx in indices:
                light_curve = X_real[idx]
                true_label = y_real[idx]
                
                # Prepare input tensor (add batch dimension)
                input_tensor = torch.from_numpy(light_curve).float().unsqueeze(0).to(device)
                
                # Make prediction (convert logits to probabilities)
                with torch.no_grad():
                    logits = model(input_tensor).item()
                    probability = torch.sigmoid(torch.tensor(logits)).item()
                
                predictions.append(probability)
            
            # Calculate statistics
            min_prob = np.min(predictions)
            max_prob = np.max(predictions)
            avg_prob = np.mean(predictions)
            std_prob = np.std(predictions)
            variation = max_prob - min_prob
            
            print(f"   Range: {min_prob:.4f} - {max_prob:.4f}")
            print(f"   Average: {avg_prob:.4f}")
            print(f"   Std Dev: {std_prob:.4f}")
            print(f"   Variation: {variation:.4f}")
            
            # Assess sensitivity
            if variation > 0.1:
                print(f"   ✅ GOOD sensitivity: {variation:.4f}")
            elif variation > 0.05:
                print(f"   ⚠️  MODERATE sensitivity: {variation:.4f}")
            else:
                print(f"   ❌ POOR sensitivity: {variation:.4f}")
        
        print(f"\n🎯 Summary:")
        print(f"   Model trained for {checkpoint.get('total_epochs_trained', 'unknown')} epochs")
        print(f"   Best validation loss: {checkpoint.get('best_val_loss', 'unknown'):.4f}")
        print(f"   Early stopping: {'Yes' if checkpoint.get('total_epochs_trained', 0) < 50 else 'No'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing model: {e}")
        return False

if __name__ == "__main__":
    test_model_sensitivity()
