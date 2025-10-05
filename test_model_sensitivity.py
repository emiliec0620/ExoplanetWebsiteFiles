#!/usr/bin/env python3
"""
Test script to check model sensitivity to different input parameters.
"""

import numpy as np
import torch
import sys
import os

# Add the current directory to Python path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from test_predict_exoplanet import predict_exoplanet

def test_parameter_sensitivity():
    """Test how the model responds to different parameter values."""
    
    print("🧪 Testing Model Parameter Sensitivity")
    print("=" * 50)
    
    # Base parameters (from your example)
    base_params = {
        'ra': "19h44m00.94s",
        'dec': "-47d33m43.31s", 
        'pmra': -3.457,
        'pmdec': -100.891,
        'transit_midpoint': 2458656.668478,
        'orbital_period': 6.443866,
        'transit_duration': 1.930945,
        'transit_depth': 1121.951228,
        'planet_radius': 2.460344,
        'planet_insolation': 60.831305,
        'planet_temp': 712.281284,
        'tess_mag': 10.005900,
        'stellar_distance': 68.072600,
        'stellar_temp': 4803.000000,
        'stellar_logg': 4.520790,
        'stellar_radius': 0.737189
    }
    
    # Test different orbital periods
    print("\n📊 Testing Different Orbital Periods:")
    periods = [1.0, 3.0, 6.4, 10.0, 20.0, 50.0]
    for period in periods:
        params = base_params.copy()
        params['orbital_period'] = period
        prob, conf, _ = predict_exoplanet(**params)
        if prob is not None:
            print(f"  Period: {period:4.1f} days -> Probability: {prob:.4f} ({prob*100:.2f}%)")
        else:
            print(f"  Period: {period:4.1f} days -> ERROR: Could not make prediction")
    
    # Test different transit depths
    print("\n📊 Testing Different Transit Depths:")
    depths = [100, 500, 1000, 2000, 5000, 10000]  # ppm
    for depth in depths:
        params = base_params.copy()
        params['transit_depth'] = depth
        prob, conf, _ = predict_exoplanet(**params)
        if prob is not None:
            print(f"  Depth: {depth:5d} ppm -> Probability: {prob:.4f} ({prob*100:.2f}%)")
        else:
            print(f"  Depth: {depth:5d} ppm -> ERROR: Could not make prediction")
    
    # Test different planet radii
    print("\n📊 Testing Different Planet Radii:")
    radii = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]  # R_Earth
    for radius in radii:
        params = base_params.copy()
        params['planet_radius'] = radius
        prob, conf, _ = predict_exoplanet(**params)
        if prob is not None:
            print(f"  Radius: {radius:4.1f} R_Earth -> Probability: {prob:.4f} ({prob*100:.2f}%)")
        else:
            print(f"  Radius: {radius:4.1f} R_Earth -> ERROR: Could not make prediction")
    
    # Test different stellar temperatures
    print("\n📊 Testing Different Stellar Temperatures:")
    temps = [3000, 4000, 5000, 6000, 7000, 8000]  # K
    for temp in temps:
        params = base_params.copy()
        params['stellar_temp'] = temp
        prob, conf, _ = predict_exoplanet(**params)
        if prob is not None:
            print(f"  Temperature: {temp:4d} K -> Probability: {prob:.4f} ({prob*100:.2f}%)")
        else:
            print(f"  Temperature: {temp:4d} K -> ERROR: Could not make prediction")
    
    # Test extreme cases
    print("\n📊 Testing Extreme Cases:")
    
    # Very short period (likely false positive)
    params = base_params.copy()
    params['orbital_period'] = 0.5
    params['transit_depth'] = 50000  # Very deep
    prob, conf, _ = predict_exoplanet(**params)
    if prob is not None:
        print(f"  Short period + deep transit -> Probability: {prob:.4f} ({prob*100:.2f}%)")
    else:
        print(f"  Short period + deep transit -> ERROR: Could not make prediction")
    
    # Very long period (likely real planet)
    params = base_params.copy()
    params['orbital_period'] = 100.0
    params['transit_depth'] = 200  # Shallow
    prob, conf, _ = predict_exoplanet(**params)
    if prob is not None:
        print(f"  Long period + shallow transit -> Probability: {prob:.4f} ({prob*100:.2f}%)")
    else:
        print(f"  Long period + shallow transit -> ERROR: Could not make prediction")
    
    # No transit signal
    params = base_params.copy()
    params['transit_depth'] = 0
    prob, conf, _ = predict_exoplanet(**params)
    if prob is not None:
        print(f"  No transit signal -> Probability: {prob:.4f} ({prob*100:.2f}%)")
    else:
        print(f"  No transit signal -> ERROR: Could not make prediction")

if __name__ == "__main__":
    test_parameter_sensitivity()
