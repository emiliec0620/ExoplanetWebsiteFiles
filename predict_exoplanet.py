import torch
import numpy as np
import re
from train_python_model import TransitCNN

def parse_sexagesimal_coord(coord_str):
    """Parse sexagesimal coordinates (RA or Dec) to decimal degrees."""
    # Remove spaces and handle different formats
    coord_str = coord_str.strip()
    
    # For RA (hours:minutes:seconds)
    if 'h' in coord_str and 'm' in coord_str and 's' in coord_str:
        # Format: 19h44m00.94s
        parts = re.split(r'[hms]', coord_str)
        hours = float(parts[0])
        minutes = float(parts[1])
        seconds = float(parts[2])
        # Convert to decimal degrees
        decimal_degrees = (hours + minutes/60 + seconds/3600) * 15
        return decimal_degrees
    
    # For Dec (degrees:arcminutes:arcseconds)
    elif 'd' in coord_str and 'm' in coord_str:
        # Format: -47d33m43.31s
        parts = re.split(r'[dms]', coord_str)
        degrees = float(parts[0])
        arcminutes = float(parts[1])
        arcseconds = float(parts[2])
        # Convert to decimal degrees
        decimal_degrees = degrees + arcminutes/60 + arcseconds/3600
        return decimal_degrees
    
    return float(coord_str)

def parse_uncertainty_value(value_str):
    """Parse values with uncertainties (e.g., '10.0059±0.006' -> 10.0059)."""
    if '±' in value_str:
        return float(value_str.split('±')[0])
    return float(value_str)

def create_light_curve_from_params(params):
    """
    Create a synthetic light curve based on exoplanet parameters.
    This simulates what TESS would observe for the given parameters.
    """
    # Extract key parameters
    orbital_period = params['orbital_period']  # days
    transit_duration = params['transit_duration']  # hours
    transit_depth = params['transit_depth'] / 1e6  # Convert ppm to fraction
    planet_radius = params['planet_radius']  # R_Earth
    stellar_radius = params['stellar_radius']  # R_Sun
    
    # TESS observation parameters
    observation_duration = 27.4  # days (TESS sector duration)
    cadence = 0.020833  # days (30 minutes)
    
    # Create time array
    time_points = int(observation_duration / cadence)
    time = np.linspace(0, observation_duration, time_points)
    
    # Create baseline flux
    flux = np.ones_like(time)
    
    # Add transit signals
    if orbital_period > 0 and transit_duration > 0:
        # Calculate number of transits in observation period
        num_transits = int(observation_duration / orbital_period)
        
        for i in range(num_transits):
            # Transit center time
            transit_center = (i + 0.5) * orbital_period
            
            # Transit duration in days
            transit_duration_days = transit_duration / 24
            
            # Transit start and end
            transit_start = transit_center - transit_duration_days / 2
            transit_end = transit_center + transit_duration_days / 2
            
            # Find indices for transit
            transit_mask = (time >= transit_start) & (time <= transit_end)
            
            # Apply transit depth
            flux[transit_mask] -= transit_depth
    
    # Add realistic noise
    noise_level = 0.001  # 0.1% noise
    flux += np.random.normal(0, noise_level, len(flux))
    
    # Normalize
    flux = (flux - np.mean(flux)) / np.std(flux)
    
    # Resize to match model input (2048 points)
    if len(flux) > 2048:
        # Downsample
        indices = np.linspace(0, len(flux)-1, 2048, dtype=int)
        flux = flux[indices]
    elif len(flux) < 2048:
        # Upsample with interpolation
        from scipy.interpolate import interp1d
        f = interp1d(np.linspace(0, 1, len(flux)), flux, kind='linear')
        flux = f(np.linspace(0, 1, 2048))
    
    return flux.astype(np.float32)

def predict_exoplanet(ra, dec, pmra, pmdec, transit_midpoint, orbital_period, 
                     transit_duration, transit_depth, planet_radius, 
                     planet_insolation, planet_temp, tess_mag, stellar_distance,
                     stellar_temp, stellar_logg, stellar_radius, model_path="transit_cnn_model.pt"):
    """
    Predict if a planet is an exoplanet based on the given parameters.
    
    Parameters:
    - ra: Right Ascension in sexagesimal format (e.g., "19h44m00.94s")
    - dec: Declination in sexagesimal format (e.g., "-47d33m43.31s")
    - pmra: Proper motion in RA [mas/yr]
    - pmdec: Proper motion in Dec [mas/yr]
    - transit_midpoint: Planet transit midpoint [BJD]
    - orbital_period: Planet orbital period [days]
    - transit_duration: Planet transit duration [hours]
    - transit_depth: Planet transit depth [ppm]
    - planet_radius: Planet radius [R_Earth]
    - planet_insolation: Planet insolation [Earth flux]
    - planet_temp: Planet equilibrium temperature [K]
    - tess_mag: TESS magnitude
    - stellar_distance: Stellar distance [pc]
    - stellar_temp: Stellar effective temperature [K]
    - stellar_logg: Stellar log(g) [cm/s**2]
    - stellar_radius: Stellar radius [R_Sun]
    - model_path: Path to the trained model file
    
    Returns:
    - prediction: Probability that this is an exoplanet (0-1)
    - confidence: Confidence level of the prediction
    """
    
    # Parse input parameters
    try:
        ra_decimal = parse_sexagesimal_coord(ra)
        dec_decimal = parse_sexagesimal_coord(dec)
        
        # Parse values with uncertainties
        pmra_val = parse_uncertainty_value(str(pmra))
        pmdec_val = parse_uncertainty_value(str(pmdec))
        transit_midpoint_val = parse_uncertainty_value(str(transit_midpoint))
        orbital_period_val = parse_uncertainty_value(str(orbital_period))
        transit_duration_val = parse_uncertainty_value(str(transit_duration))
        transit_depth_val = parse_uncertainty_value(str(transit_depth))
        planet_radius_val = parse_uncertainty_value(str(planet_radius))
        planet_insolation_val = parse_uncertainty_value(str(planet_insolation))
        planet_temp_val = parse_uncertainty_value(str(planet_temp))
        tess_mag_val = parse_uncertainty_value(str(tess_mag))
        stellar_distance_val = parse_uncertainty_value(str(stellar_distance))
        stellar_temp_val = parse_uncertainty_value(str(stellar_temp))
        stellar_logg_val = parse_uncertainty_value(str(stellar_logg))
        stellar_radius_val = parse_uncertainty_value(str(stellar_radius))
        
    except Exception as e:
        print(f"Error parsing input parameters: {e}")
        return None, None
    
    # Create parameter dictionary
    params = {
        'ra': ra_decimal,
        'dec': dec_decimal,
        'pmra': pmra_val,
        'pmdec': pmdec_val,
        'transit_midpoint': transit_midpoint_val,
        'orbital_period': orbital_period_val,
        'transit_duration': transit_duration_val,
        'transit_depth': transit_depth_val,
        'planet_radius': planet_radius_val,
        'planet_insolation': planet_insolation_val,
        'planet_temp': planet_temp_val,
        'tess_mag': tess_mag_val,
        'stellar_distance': stellar_distance_val,
        'stellar_temp': stellar_temp_val,
        'stellar_logg': stellar_logg_val,
        'stellar_radius': stellar_radius_val
    }
    
    # Create synthetic light curve
    light_curve = create_light_curve_from_params(params)
    
    # Load the trained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # Load model with weights_only=False for custom classes
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model = checkpoint['model_architecture']
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        # Prepare input tensor
        input_tensor = torch.from_numpy(light_curve).float().unsqueeze(0).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            prediction = model(input_tensor)
            probability = prediction.item()
        
        # Determine confidence level
        if probability > 0.8 or probability < 0.2:
            confidence = "High"
        elif probability > 0.6 or probability < 0.4:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        return probability, confidence, params
        
    except FileNotFoundError:
        print(f"Model file {model_path} not found. Please train the model first.")
        return None, None, None
    except Exception as e:
        print(f"Error loading model or making prediction: {e}")
        return None, None, None

def main():
    """Example usage of the exoplanet prediction function."""
    
    # Example input parameters (from your data)
    ra = "19h44m00.94s"
    dec = "-47d33m43.31s"
    pmra = "-3.457±0.124"
    pmdec = "-100.891±0.072"
    transit_midpoint = "2458656.668478±0.0011852143"
    orbital_period = "6.44386574740406±0.00001645013"
    transit_duration = "1.93094488736779±0.30149648"
    transit_depth = "1121.95122750953±76.157646"
    planet_radius = "2.46034402739473±1.638728"
    planet_insolation = "60.8313050283028"
    planet_temp = "712.281284080803"
    tess_mag = "10.0059±0.006"
    stellar_distance = "68.0726±0.362"
    stellar_temp = "4803±178"
    stellar_logg = "4.52079±2.00348"
    stellar_radius = "0.737188994884491±0.0567035"
    
    print("Exoplanet Prediction System")
    print("=" * 50)
    print(f"Input Parameters:")
    print(f"RA: {ra}")
    print(f"Dec: {dec}")
    print(f"Orbital Period: {orbital_period} days")
    print(f"Transit Duration: {transit_duration} hours")
    print(f"Transit Depth: {transit_depth} ppm")
    print(f"Planet Radius: {planet_radius} R_Earth")
    print(f"Stellar Temperature: {stellar_temp} K")
    print()
    
    # Make prediction
    probability, confidence, parsed_params = predict_exoplanet(
        ra, dec, pmra, pmdec, transit_midpoint, orbital_period,
        transit_duration, transit_depth, planet_radius, planet_insolation,
        planet_temp, tess_mag, stellar_distance, stellar_temp, 
        stellar_logg, stellar_radius
    )
    
    if probability is not None:
        print("Prediction Results:")
        print(f"Exoplanet Probability: {probability:.4f} ({probability*100:.2f}%)")
        print(f"Confidence Level: {confidence}")
        
        if probability > 0.5:
            print("Prediction: LIKELY EXOPLANET")
        else:
            print("Prediction: LIKELY NOT AN EXOPLANET")
        
        print(f"\nParsed Parameters:")
        for key, value in parsed_params.items():
            print(f"{key}: {value:.6f}")
    else:
        print("Prediction failed. Please check the model file and input parameters.")

if __name__ == "__main__":
    main()
