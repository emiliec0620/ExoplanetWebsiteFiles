#!/usr/bin/env python3
"""
Real TESS Data Exoplanet Prediction System
Uses actual preprocessed TESS light curves for realistic predictions.
"""

import torch
import numpy as np
import sys
import os
from train_python_model import TransitCNN, load_preprocessed_data

def predict_exoplanet_from_real_data(tic_id=None, model_path="transit_cnn_model.pt"):
    """
    Predict exoplanet probability using real TESS light curve data.
    
    Args:
        tic_id: TIC ID to predict (if None, uses random sample)
        model_path: Path to trained model
    
    Returns:
        probability, confidence, tic_id_used
    """
    try:
        # Load the model
        device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model = checkpoint['model_architecture']
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        # Load real TESS data
        X_real, y_real = load_preprocessed_data("preprocessed_data")
        
        if tic_id is None:
            # Use random sample
            idx = np.random.randint(0, len(X_real))
        else:
            # Find TIC ID in data (this would need to be implemented with metadata)
            # For now, use random sample
            idx = np.random.randint(0, len(X_real))
        
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
        
        return probability, confidence, idx
        
    except Exception as e:
        print(f"Error making prediction: {e}")
        return None, None, None

def batch_predict_real_data(num_samples=10, model_path="transit_cnn_model.pt"):
    """
    Make batch predictions on real TESS data.
    """
    print("🔬 Real TESS Data Exoplanet Prediction System")
    print("=" * 50)
    
    try:
        # Load the model
        device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model = checkpoint['model_architecture']
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        print(f"✅ Model loaded on {device}")
        print(f"📊 Trained for {checkpoint.get('total_epochs_trained', 'unknown')} epochs")
        print(f"📈 Best validation loss: {checkpoint.get('best_val_loss', 'unknown'):.4f}")
        
        # Load real TESS data
        X_real, y_real = load_preprocessed_data("preprocessed_data")
        print(f"✅ Loaded {len(X_real)} real TESS light curves")
        
        # Select random samples
        np.random.seed(42)
        indices = np.random.choice(len(X_real), min(num_samples, len(X_real)), replace=False)
        
        print(f"\n🧪 Making predictions on {len(indices)} real TESS light curves:")
        print("-" * 50)
        
        predictions = []
        exoplanet_count = 0
        
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
            if probability > 0.5:
                exoplanet_count += 1
            
            # Check accuracy
            correct = "✅" if (probability > 0.5) == (true_label > 0.5) else "❌"
            
            print(f"Sample {i+1:2d}: {probability:.4f} ({probability*100:.2f}%) - {prediction} - {confidence} {correct}")
            
            predictions.append(probability)
        
        # Summary
        print("\n" + "=" * 50)
        print("SUMMARY:")
        print(f"Total samples: {len(predictions)}")
        print(f"Predicted exoplanets: {exoplanet_count}")
        print(f"Predicted non-exoplanets: {len(predictions) - exoplanet_count}")
        print(f"Average probability: {np.mean(predictions):.4f}")
        print(f"Probability range: {np.min(predictions):.4f} - {np.max(predictions):.4f}")
        print(f"Variation: {np.max(predictions) - np.min(predictions):.4f}")
        
        return predictions
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    # Test with different numbers of samples
    print("Testing with 5 samples:")
    batch_predict_real_data(5)
    
    print("\n" + "="*60 + "\n")
    
    print("Testing with 20 samples:")
    batch_predict_real_data(20)
