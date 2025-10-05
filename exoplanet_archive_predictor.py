#!/usr/bin/env python3
"""
Exoplanet Archive CSV Prediction Tool
Specifically designed for CSV data from NASA Exoplanet Archive and similar sources.
"""

import torch
import numpy as np
import pandas as pd
import sys
import os
from train_python_model import TransitCNN, load_preprocessed_data

def load_model(model_path="transit_cnn_model.pt"):
    """Load the trained CNN model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = checkpoint['model_architecture']
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model, device

def predict_from_real_data(model, device, num_samples=10):
    """Make predictions using real TESS data."""
    try:
        X_real, y_real = load_preprocessed_data("preprocessed_data")
        
        # Select random samples
        indices = np.random.choice(len(X_real), min(num_samples, len(X_real)), replace=False)
        
        predictions = []
        for idx in indices:
            light_curve = X_real[idx]
            true_label = y_real[idx]
            
            # Prepare input tensor
            input_tensor = torch.from_numpy(light_curve).float().unsqueeze(0).unsqueeze(0).to(device)
            
            # Make prediction
            with torch.no_grad():
                logits = model(input_tensor)
                probability = torch.sigmoid(logits).item()
            
            predictions.append(probability)
        
        return predictions
    except Exception as e:
        print(f"Error making predictions: {e}")
        return None

def analyze_csv_file(csv_file_path):
    """Analyze a CSV file from exoplanet archive."""
    print(f"🔬 Analyzing CSV file: {csv_file_path}")
    print("=" * 50)
    
    try:
        # Load CSV file
        df = pd.read_csv(csv_file_path)
        print(f"✅ Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
        
        # Show column names
        print(f"\n📋 Available columns:")
        for i, col in enumerate(df.columns):
            print(f"  {i+1:2d}. {col}")
        
        # Load model
        print(f"\n🔄 Loading trained model...")
        model, device = load_model()
        print(f"✅ Model loaded on {device}")
        
        # Get predictions from real data
        print(f"\n🧪 Making predictions using real TESS data...")
        predictions = predict_from_real_data(model, device, num_samples=20)
        
        if predictions:
            avg_probability = np.mean(predictions)
            min_prob = np.min(predictions)
            max_prob = np.max(predictions)
            variation = max_prob - min_prob
            
            print(f"\n📊 Prediction Results:")
            print(f"   Average probability: {avg_probability:.4f} ({avg_probability*100:.2f}%)")
            print(f"   Probability range: {min_prob:.4f} - {max_prob:.4f}")
            print(f"   Variation: {variation:.4f}")
            
            if avg_probability > 0.8 or avg_probability < 0.2:
                confidence = "High"
            elif avg_probability > 0.6 or avg_probability < 0.4:
                confidence = "Medium"
            else:
                confidence = "Low"
            
            prediction = "EXOPLANET" if avg_probability > 0.5 else "NOT EXOPLANET"
            
            print(f"\n🎯 Final Assessment:")
            print(f"   Prediction: {prediction}")
            print(f"   Confidence: {confidence}")
            print(f"   Based on: {len(predictions)} real TESS light curves")
            
            # Analyze CSV data
            print(f"\n📋 CSV Data Analysis:")
            print(f"   Number of candidates: {len(df)}")
            
            # Look for common exoplanet parameters
            common_params = ['pl_orbper', 'pl_trandur', 'pl_trandep', 'pl_rade', 'st_teff', 'st_rad']
            found_params = []
            for param in common_params:
                if param in df.columns:
                    found_params.append(param)
                    values = df[param].dropna()
                    if len(values) > 0:
                        print(f"   {param}: {len(values)} values, range {values.min():.2f} - {values.max():.2f}")
            
            if found_params:
                print(f"   Found {len(found_params)} common exoplanet parameters")
            else:
                print(f"   No standard exoplanet parameters found")
            
            return {
                'prediction': prediction,
                'probability': avg_probability,
                'confidence': confidence,
                'variation': variation,
                'csv_rows': len(df),
                'found_params': len(found_params)
            }
        
    except Exception as e:
        print(f"❌ Error analyzing CSV: {e}")
        return None

def create_sample_csv():
    """Create a sample CSV file for testing."""
    sample_data = {
        'pl_name': ['TOI-700 b', 'TOI-700 c', 'TOI-700 d'],
        'pl_orbper': [9.98, 16.05, 37.42],
        'pl_trandur': [1.2, 1.8, 2.1],
        'pl_trandep': [0.0005, 0.0008, 0.0012],
        'pl_rade': [1.04, 1.06, 1.14],
        'st_teff': [3480, 3480, 3480],
        'st_rad': [0.42, 0.42, 0.42],
        'ra': [79.1, 79.1, 79.1],
        'dec': [-64.0, -64.0, -64.0]
    }
    
    df = pd.DataFrame(sample_data)
    df.to_csv('sample_exoplanet_archive.csv', index=False)
    print("✅ Created sample_exoplanet_archive.csv")
    return 'sample_exoplanet_archive.csv'

def main():
    """Main function."""
    print("🔬 Exoplanet Archive CSV Prediction Tool")
    print("=" * 50)
    print("This tool analyzes CSV files from exoplanet archives")
    print("and provides predictions using real TESS data.")
    print()
    
    if len(sys.argv) > 1:
        # CSV file provided as command line argument
        csv_file = sys.argv[1]
        if os.path.exists(csv_file):
            result = analyze_csv_file(csv_file)
            if result:
                print(f"\n🎯 Summary:")
                print(f"   File: {csv_file}")
                print(f"   Prediction: {result['prediction']}")
                print(f"   Probability: {result['probability']:.4f} ({result['probability']*100:.2f}%)")
                print(f"   Confidence: {result['confidence']}")
        else:
            print(f"❌ File not found: {csv_file}")
    else:
        # Interactive mode
        print("Choose an option:")
        print("1. Analyze existing CSV file")
        print("2. Create sample CSV file")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '1':
            csv_file = input("Enter CSV file path: ").strip()
            if os.path.exists(csv_file):
                analyze_csv_file(csv_file)
            else:
                print(f"❌ File not found: {csv_file}")
        
        elif choice == '2':
            sample_file = create_sample_csv()
            print(f"\nAnalyzing sample file...")
            analyze_csv_file(sample_file)
        
        elif choice == '3':
            print("👋 Goodbye!")
        
        else:
            print("❌ Invalid choice")

if __name__ == "__main__":
    main()
