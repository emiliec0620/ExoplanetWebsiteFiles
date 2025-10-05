#!/usr/bin/env python3
"""
Command-line exoplanet prediction tool.
Usage: python exoplanet_predictor.py <input_file.csv> [model_path]
"""

import sys
import csv
import torch
import numpy as np
from test_predict_exoplanet import predict_exoplanet
from train_python_model import TransitCNN

def load_csv_data(filename):
    """Load exoplanet data from CSV file."""
    data = []
    with open(filename, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)
    return data

def predict_from_csv(csv_file, model_path="transit_cnn_model.pt"):
    """Predict exoplanets from CSV data."""
    try:
        data = load_csv_data(csv_file)
        print(f"Loaded {len(data)} records from {csv_file}")
        print("=" * 80)
        
        results = []
        
        for i, row in enumerate(data):
            print(f"\nRecord {i+1}:")
            print("-" * 40)
            
            # Extract parameters
            ra = row.get('RA [sexagesimal]', '')
            dec = row.get('Dec [sexagesimal]', '')
            pmra = row.get('PMRA [mas/yr]', '0')
            pmdec = row.get('PMDec [mas/yr]', '0')
            transit_midpoint = row.get('Planet Transit Midpoint [BJD]', '0')
            orbital_period = row.get('Planet Orbital Period [days]', '0')
            transit_duration = row.get('Planet Transit Duration [hours]', '0')
            transit_depth = row.get('Planet Transit Depth [ppm]', '0')
            planet_radius = row.get('Planet Radius [R_Earth]', '0')
            planet_insolation = row.get('Planet Insolation [Earth flux]', '0')
            planet_temp = row.get('Planet Equilibrium Temperature [K]', '0')
            tess_mag = row.get('TESS Magnitude', '0')
            stellar_distance = row.get('Stellar Distance [pc]', '0')
            stellar_temp = row.get('Stellar Effective Temperature [K]', '0')
            stellar_logg = row.get('Stellar log(g) [cm/s**2]', '0')
            stellar_radius = row.get('Stellar Radius [R_Sun]', '0')
            
            # Make prediction
            probability, confidence, parsed_params = predict_exoplanet(
                ra, dec, pmra, pmdec, transit_midpoint, orbital_period,
                transit_duration, transit_depth, planet_radius, planet_insolation,
                planet_temp, tess_mag, stellar_distance, stellar_temp,
                stellar_logg, stellar_radius, model_path
            )
            
            if probability is not None:
                result = {
                    'record': i+1,
                    'probability': probability,
                    'confidence': confidence,
                    'prediction': 'EXOPLANET' if probability > 0.5 else 'NOT EXOPLANET',
                    'ra': ra,
                    'dec': dec,
                    'orbital_period': orbital_period
                }
                results.append(result)
                
                print(f"Exoplanet Probability: {probability:.4f} ({probability*100:.2f}%)")
                print(f"Confidence: {confidence}")
                print(f"Prediction: {result['prediction']}")
            else:
                print("Prediction failed for this record")
        
        # Summary
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        
        exoplanet_count = sum(1 for r in results if r['prediction'] == 'EXOPLANET')
        total_count = len(results)
        
        print(f"Total records processed: {total_count}")
        print(f"Predicted exoplanets: {exoplanet_count}")
        print(f"Predicted non-exoplanets: {total_count - exoplanet_count}")
        
        if exoplanet_count > 0:
            print(f"\nExoplanet candidates:")
            for result in results:
                if result['prediction'] == 'EXOPLANET':
                    print(f"  Record {result['record']}: {result['probability']:.4f} "
                          f"({result['confidence']} confidence) - "
                          f"RA: {result['ra']}, Dec: {result['dec']}")
        
        return results
        
    except FileNotFoundError:
        print(f"Error: File {csv_file} not found")
        return None
    except Exception as e:
        print(f"Error processing data: {e}")
        return None

def main():
    if len(sys.argv) < 2:
        print("Usage: python exoplanet_predictor.py <input_file.csv> [model_path]")
        print("\nCSV file should have columns:")
        print("- RA [sexagesimal]")
        print("- Dec [sexagesimal]")
        print("- PMRA [mas/yr]")
        print("- PMDec [mas/yr]")
        print("- Planet Transit Midpoint [BJD]")
        print("- Planet Orbital Period [days]")
        print("- Planet Transit Duration [hours]")
        print("- Planet Transit Depth [ppm]")
        print("- Planet Radius [R_Earth]")
        print("- Planet Insolation [Earth flux]")
        print("- Planet Equilibrium Temperature [K]")
        print("- TESS Magnitude")
        print("- Stellar Distance [pc]")
        print("- Stellar Effective Temperature [K]")
        print("- Stellar log(g) [cm/s**2]")
        print("- Stellar Radius [R_Sun]")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else "transit_cnn_model.pt"
    
    print("Exoplanet Prediction Tool")
    print("=" * 50)
    print(f"Input file: {csv_file}")
    print(f"Model file: {model_path}")
    print()
    
    results = predict_from_csv(csv_file, model_path)
    
    if results is not None:
        print(f"\nResults saved to memory. Processed {len(results)} records.")

if __name__ == "__main__":
    main()
