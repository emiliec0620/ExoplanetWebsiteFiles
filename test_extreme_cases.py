#!/usr/bin/env python3
"""
Simple test to check if the model responds to different inputs.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from test_predict_exoplanet import predict_exoplanet

def test_extreme_cases():
    """Test extreme cases to see if model responds differently."""
    
    print("🧪 Testing Extreme Cases")
    print("=" * 40)
    
    # Test 1: Normal exoplanet parameters
    print("\n1. Normal Exoplanet Parameters:")
    prob1, conf1, _ = predict_exoplanet(
        ra="19h44m00.94s",
        dec="-47d33m43.31s",
        pmra=-3.457,
        pmdec=-100.891,
        transit_midpoint=2458656.668478,
        orbital_period=6.443866,
        transit_duration=1.930945,
        transit_depth=1121.951228,
        planet_radius=2.460344,
        planet_insolation=60.831305,
        planet_temp=712.281284,
        tess_mag=10.005900,
        stellar_distance=68.072600,
        stellar_temp=4803.000000,
        stellar_logg=4.520790,
        stellar_radius=0.737189
    )
    if prob1 is not None:
        print(f"   Probability: {prob1:.4f} ({prob1*100:.2f}%)")
    else:
        print("   ERROR: Could not make prediction")
    
    # Test 2: No transit signal (should be low probability)
    print("\n2. No Transit Signal:")
    prob2, conf2, _ = predict_exoplanet(
        ra="19h44m00.94s",
        dec="-47d33m43.31s",
        pmra=-3.457,
        pmdec=-100.891,
        transit_midpoint=2458656.668478,
        orbital_period=6.443866,
        transit_duration=1.930945,
        transit_depth=0,  # No transit
        planet_radius=2.460344,
        planet_insolation=60.831305,
        planet_temp=712.281284,
        tess_mag=10.005900,
        stellar_distance=68.072600,
        stellar_temp=4803.000000,
        stellar_logg=4.520790,
        stellar_radius=0.737189
    )
    if prob2 is not None:
        print(f"   Probability: {prob2:.4f} ({prob2*100:.2f}%)")
    else:
        print("   ERROR: Could not make prediction")
    
    # Test 3: Very deep transit (likely false positive)
    print("\n3. Very Deep Transit (False Positive):")
    prob3, conf3, _ = predict_exoplanet(
        ra="19h44m00.94s",
        dec="-47d33m43.31s",
        pmra=-3.457,
        pmdec=-100.891,
        transit_midpoint=2458656.668478,
        orbital_period=0.5,  # Very short period
        transit_duration=1.0,
        transit_depth=50000,  # Very deep transit
        planet_radius=20.0,  # Very large planet
        planet_insolation=60.831305,
        planet_temp=712.281284,
        tess_mag=10.005900,
        stellar_distance=68.072600,
        stellar_temp=3000,  # Cool star
        stellar_logg=4.520790,
        stellar_radius=0.737189
    )
    if prob3 is not None:
        print(f"   Probability: {prob3:.4f} ({prob3*100:.2f}%)")
    else:
        print("   ERROR: Could not make prediction")
    
    # Test 4: Very shallow transit (likely real planet)
    print("\n4. Very Shallow Transit (Real Planet):")
    prob4, conf4, _ = predict_exoplanet(
        ra="19h44m00.94s",
        dec="-47d33m43.31s",
        pmra=-3.457,
        pmdec=-100.891,
        transit_midpoint=2458656.668478,
        orbital_period=50.0,  # Long period
        transit_duration=3.0,
        transit_depth=50,  # Very shallow transit
        planet_radius=0.5,  # Small planet
        planet_insolation=60.831305,
        planet_temp=712.281284,
        tess_mag=8.0,  # Bright star
        stellar_distance=68.072600,
        stellar_temp=6000,  # Hot star
        stellar_logg=4.520790,
        stellar_radius=0.737189
    )
    if prob4 is not None:
        print(f"   Probability: {prob4:.4f} ({prob4*100:.2f}%)")
    else:
        print("   ERROR: Could not make prediction")
    
    # Summary
    print("\n" + "=" * 40)
    print("SUMMARY:")
    if prob1 is not None and prob2 is not None and prob3 is not None and prob4 is not None:
        print(f"Normal case:     {prob1:.4f}")
        print(f"No transit:      {prob2:.4f}")
        print(f"False positive:  {prob3:.4f}")
        print(f"Real planet:     {prob4:.4f}")
        
        # Check if there's variation
        probabilities = [prob1, prob2, prob3, prob4]
        min_prob = min(probabilities)
        max_prob = max(probabilities)
        variation = max_prob - min_prob
        
        print(f"\nVariation range: {variation:.4f}")
        if variation < 0.01:
            print("⚠️  WARNING: Very little variation - model may not be sensitive to inputs")
        else:
            print("✅ Model shows sensitivity to different inputs")
    else:
        print("❌ Some predictions failed")

if __name__ == "__main__":
    test_extreme_cases()
