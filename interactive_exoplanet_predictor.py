#!/usr/bin/env python3
"""
Interactive Exoplanet Prediction Tool
Allows users to paste CSV data from exoplanet archives and get predictions.
Now uses Random Forest model for better sensitivity and performance.
"""

import numpy as np
import pandas as pd
import sys
import os
import pickle
import json
from io import StringIO
from scipy import interpolate
from preprocess_data import load_preprocessed_data
from random_forest_exoplanet_model import extract_features_vectorized

class ExoplanetPredictor:
    def __init__(self, model_path="random_forest_exoplanet_model.pkl"):
        """Initialize the predictor with the trained Random Forest model."""
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.real_data = None
        self.load_model()
        self.load_real_data()
    
    def load_model(self):
        """Load the trained Random Forest model."""
        try:
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            auc_score = model_data['auc_score']
            
            print(f"✅ Random Forest model loaded successfully")
            print(f"📊 Model AUC Score: {auc_score:.4f}")
            print(f"🔍 Number of features: {len(self.feature_names)}")
            print(f"🌲 Number of trees: {self.model.n_estimators}")
            
        except Exception as e:
            print(f"❌ Error loading Random Forest model: {e}")
            print("Please train the Random Forest model first by running random_forest_exoplanet_model.py")
            sys.exit(1)
    
    def load_real_data(self):
        """Load real TESS data for reference."""
        try:
            X_real, y_real = load_preprocessed_data("preprocessed_data")
            self.real_data = (X_real, y_real)
            print(f"✅ Loaded {len(X_real)} real TESS light curves for reference")
        except Exception as e:
            print(f"⚠️  Warning: Could not load real data: {e}")
            self.real_data = None
    
    def predict_from_real_data(self, num_samples=1):
        """Make predictions using real TESS data."""
        if self.real_data is None:
            print("❌ No real data available")
            return None
        
        X_real, y_real = self.real_data
        
        # Select random samples
        indices = np.random.choice(len(X_real), min(num_samples, len(X_real)), replace=False)
        
        results = []
        for idx in indices:
            light_curve = X_real[idx]
            true_label = y_real[idx]
            
            # Extract features and make prediction using vectorized method
            features = extract_features_vectorized(np.array([light_curve]))
            features_scaled = self.scaler.transform(features)
            probability = self.model.predict_proba(features_scaled)[0, 1]
            
            # Determine confidence level
            if probability > 0.8 or probability < 0.2:
                confidence = "High"
            elif probability > 0.6 or probability < 0.4:
                confidence = "Medium"
            else:
                confidence = "Low"
            
            prediction = "EXOPLANET" if probability > 0.5 else "NOT EXOPLANET"
            
            results.append({
                'probability': probability,
                'confidence': confidence,
                'prediction': prediction,
                'true_label': "EXOPLANET" if true_label > 0.5 else "NOT EXOPLANET",
                'correct': (probability > 0.5) == (true_label > 0.5)
            })
        
        return results
    
    def parse_csv_data(self, csv_text):
        """Parse CSV data pasted by the user."""
        try:
            # Try to parse the CSV
            df = pd.read_csv(StringIO(csv_text))
            print(f"✅ Successfully parsed CSV with {len(df)} rows and {len(df.columns)} columns")
            print(f"📋 Columns: {list(df.columns)}")
            return df
        except Exception as e:
            print(f"❌ Error parsing CSV: {e}")
            return None
    
    def map_columns(self, df):
        """Map CSV columns to our prediction parameters."""
        # Common column name mappings
        column_mappings = {
            'ra': ['ra', 'RA', 'right_ascension', 'Right Ascension'],
            'dec': ['dec', 'Dec', 'declination', 'Declination'],
            'pmra': ['pmra', 'PMRA', 'pm_ra', 'proper_motion_ra'],
            'pmdec': ['pmdec', 'PMDec', 'pm_dec', 'proper_motion_dec'],
            'orbital_period': ['orbital_period', 'period', 'Period', 'orbital period', 'P'],
            'transit_duration': ['transit_duration', 'duration', 'Duration', 'transit duration', 'T14'],
            'transit_depth': ['transit_depth', 'depth', 'Depth', 'transit depth', 'delta'],
            'planet_radius': ['planet_radius', 'radius', 'Radius', 'planet radius', 'Rp'],
            'stellar_temp': ['stellar_temp', 'teff', 'Teff', 'stellar temperature', 'T_eff'],
            'stellar_radius': ['stellar_radius', 'stellar_radius', 'R_star', 'stellar radius', 'Rs'],
            'tess_mag': ['tess_mag', 'Tmag', 'tess_magnitude', 'TESS mag'],
            'stellar_distance': ['stellar_distance', 'distance', 'Distance', 'stellar distance', 'd']
        }
        
        mapped_data = {}
        for param, possible_names in column_mappings.items():
            found_column = None
            for col_name in df.columns:
                if any(name.lower() in col_name.lower() for name in possible_names):
                    found_column = col_name
                    break
            
            if found_column:
                mapped_data[param] = df[found_column].iloc[0] if len(df) > 0 else None
                print(f"✅ Mapped '{found_column}' -> {param}")
            else:
                print(f"⚠️  No column found for {param}")
                mapped_data[param] = None
        
        return mapped_data
    
    def create_synthetic_light_curve(self, df):
        """Create a synthetic light curve from exoplanet parameters in the input data."""
        try:
            print("🔍 Searching for exoplanet parameters in CSV data...")
            
            # Extract exoplanet parameters with more flexible column matching
            period_col = None
            duration_col = None
            depth_col = None
            midpoint_col = None
            
            # Look for parameter columns with flexible matching
            for col in df.columns:
                col_lower = col.lower()
                if 'period' in col_lower and ('day' in col_lower or 'orbital' in col_lower):
                    period_col = col
                elif 'duration' in col_lower and ('hour' in col_lower or 'transit' in col_lower):
                    duration_col = col
                elif 'depth' in col_lower and ('ppm' in col_lower or 'transit' in col_lower):
                    depth_col = col
                elif 'midpoint' in col_lower and ('bjd' in col_lower or 'transit' in col_lower):
                    midpoint_col = col
            
            print(f"📋 Found parameter columns:")
            print(f"   Period: {period_col}")
            print(f"   Duration: {duration_col}")
            print(f"   Depth: {depth_col}")
            print(f"   Midpoint: {midpoint_col}")
            
            if not all([period_col, duration_col, depth_col]):
                print("❌ Could not find required exoplanet parameters (period, duration, depth)")
                print(f"   Available columns: {list(df.columns)}")
                return None
            
            # Extract parameter values (handle uncertainty notation like "341.28±0.004")
            def extract_value(param_str):
                if pd.isna(param_str):
                    return None
                param_str = str(param_str).strip()
                if '±' in param_str:
                    return float(param_str.split('±')[0])
                elif ' ' in param_str:
                    # Handle cases like "341.28 0.004" 
                    return float(param_str.split()[0])
                else:
                    return float(param_str)
            
            # Extract values from the first row
            period = extract_value(df[period_col].iloc[0])
            duration_hours = extract_value(df[duration_col].iloc[0])
            depth_ppm = extract_value(df[depth_col].iloc[0])
            
            # Optional midpoint
            midpoint_bjd = None
            if midpoint_col:
                midpoint_bjd = extract_value(df[midpoint_col].iloc[0])
            
            print(f"\n📊 Extracted parameters from input data:")
            print(f"   Orbital Period: {period:.3f} days")
            print(f"   Transit Duration: {duration_hours:.3f} hours")
            print(f"   Transit Depth: {depth_ppm:.1f} ppm")
            if midpoint_bjd:
                print(f"   Transit Midpoint: {midpoint_bjd:.6f} BJD")
            
            # Validate parameters
            if not all([period, duration_hours, depth_ppm]):
                print("❌ Could not extract valid parameter values")
                return None
            
            if period <= 0 or duration_hours <= 0 or depth_ppm <= 0:
                print("❌ Invalid parameter values (must be positive)")
                return None
            
            # Create synthetic light curve
            print("   Creating synthetic light curve...")
            
            # Convert duration from hours to days
            duration_days = duration_hours / 24.0
            
            # Determine observation time based on orbital period
            # For long periods, observe longer to catch at least one transit
            if period > 30:
                total_time = period * 1.5  # Observe 1.5 periods
            else:
                total_time = max(30.0, period * 2)  # At least 30 days or 2 periods
            
            # TESS-like cadence (30 minutes)
            cadence = 0.02  # days
            time_points = int(total_time / cadence)
            
            print(f"   Observation period: {total_time:.1f} days")
            print(f"   Data points: {time_points}")
            
            time = np.linspace(0, total_time, time_points)
            flux = np.ones(time_points)  # Baseline flux
            
            # Add transits using the actual parameters from input data
            transit_depth = depth_ppm / 1e6  # Convert ppm to fractional depth
            
            # Calculate number of transits in the time period
            num_transits = int(total_time / period)
            
            print(f"   Expected transits: {num_transits}")
            
            # Use midpoint if available, otherwise start from beginning
            start_time = 0
            if midpoint_bjd:
                # Convert BJD to relative time (simplified)
                start_time = midpoint_bjd % period
            
            for i in range(num_transits + 1):  # +1 to ensure we catch all transits
                transit_time = start_time + i * period
                
                if transit_time > total_time:
                    break
                
                # Create realistic transit shape using actual duration
                transit_start = transit_time - duration_days / 2
                transit_end = transit_time + duration_days / 2
                
                # Find indices for this transit
                start_idx = max(0, np.argmin(np.abs(time - transit_start)))
                end_idx = min(len(time)-1, np.argmin(np.abs(time - transit_end)))
                
                # Create ingress/egress (realistic transit shape)
                ingress_duration = duration_days * 0.1  # 10% of total duration
                egress_duration = duration_days * 0.1
                
                for j in range(start_idx, end_idx + 1):
                    if j >= len(time):
                        break
                    
                    t_rel = time[j] - transit_time
                    
                    if abs(t_rel) <= duration_days / 2 - ingress_duration:
                        # Full transit - use actual depth from input data
                        flux[j] = 1.0 - transit_depth
                    elif abs(t_rel) <= duration_days / 2:
                        # Ingress/egress
                        if t_rel < 0:  # Ingress
                            progress = (abs(t_rel) - (duration_days / 2 - ingress_duration)) / ingress_duration
                        else:  # Egress
                            progress = (abs(t_rel) - (duration_days / 2 - ingress_duration)) / egress_duration
                        
                        flux[j] = 1.0 - transit_depth * (1.0 - progress)
            
            # Add realistic noise based on transit depth
            # Deeper transits typically have better signal-to-noise
            noise_level = min(0.01, transit_depth * 0.1)  # Noise proportional to transit depth
            noise = np.random.normal(0, noise_level, len(flux))
            flux = flux + noise
            
            print(f"   Generated {len(flux)} data points over {total_time:.1f} days")
            print(f"   Transit depth: {transit_depth*100:.3f}%")
            print(f"   Noise level: {noise_level*100:.3f}%")
            
            return flux
            
        except Exception as e:
            print(f"❌ Error creating synthetic light curve: {e}")
            return None
    
    def predict_from_csv_data(self, df):
        """Make predictions from CSV data."""
        print("\n🔬 Processing CSV light curve data...")
        
        # Map CSV columns for display
        mapped_data = self.map_columns(df)
        
        print(f"\n📋 CSV Data Analysis:")
        print("-" * 30)
        for param, value in mapped_data.items():
            if value is not None:
                print(f"{param}: {value}")
            else:
                print(f"{param}: Not found")
        
        # Extract light curve data from CSV and make prediction
        try:
            # Look for time and flux columns
            time_col = None
            flux_col = None
            
            # Common column name patterns
            time_patterns = ['time', 'Time', 'TIME', 't', 'T', 'jd', 'JD', 'bjd', 'BJD']
            flux_patterns = ['flux', 'Flux', 'FLUX', 'f', 'F', 'mag', 'Mag', 'MAG', 'brightness', 'Brightness']
            
            for col in df.columns:
                col_lower = col.lower()
                if any(pattern.lower() in col_lower for pattern in time_patterns):
                    time_col = col
                elif any(pattern.lower() in col_lower for pattern in flux_patterns):
                    flux_col = col
            
            skip_to_prediction = False
            
            if time_col is None or flux_col is None:
                print(f"⚠️ Could not find time/flux columns. Available columns: {list(df.columns)}")
                print("🔄 Attempting to create synthetic light curve from exoplanet parameters...")
                
                # Try to create synthetic light curve from parameters
                flux_resampled = self.create_synthetic_light_curve(df)
                if flux_resampled is None:
                    return None
                
                # Use synthetic light curve for prediction
                print("   Extracting features from synthetic light curve...")
                features = extract_features_vectorized(np.array([flux_resampled]))
                features_scaled = self.scaler.transform(features)
                avg_probability = self.model.predict_proba(features_scaled)[0, 1]
                
                print(f"✅ Successfully processed synthetic light curve")
                skip_to_prediction = True
            else:
                print(f"📊 Found time column: {time_col}, flux column: {flux_col}")
                
                # Extract and clean the light curve data
                print(f"   Processing {len(df)} data points...")
                time_data = pd.to_numeric(df[time_col], errors='coerce')
                flux_data = pd.to_numeric(df[flux_col], errors='coerce')
                
                # Remove NaN values
                valid_mask = ~(np.isnan(time_data) | np.isnan(flux_data))
                time_clean = time_data[valid_mask].values
                flux_clean = flux_data[valid_mask].values
                
                print(f"   Cleaned data: {len(flux_clean)} valid points")
                
                if len(flux_clean) < 100:
                    print(f"❌ Insufficient data points ({len(flux_clean)}). Need at least 100 points.")
                    return None
                else:
                    # Resample to standard length (2048 points)
                    if len(flux_clean) != 2048:
                        print(f"   Resampling from {len(flux_clean)} to 2048 points...")
                        x_new = np.linspace(0, len(flux_clean)-1, 2048)
                        f = interpolate.interp1d(np.arange(len(flux_clean)), flux_clean, kind='linear')
                        flux_resampled = f(x_new)
                    else:
                        flux_resampled = flux_clean
                    
                    # Extract features and make prediction
                    print("   Extracting features...")
                    features = extract_features_vectorized(np.array([flux_resampled]))
                    features_scaled = self.scaler.transform(features)
                    avg_probability = self.model.predict_proba(features_scaled)[0, 1]
                    
                    print(f"✅ Successfully processed CSV light curve data")
                    
        except Exception as e:
            print(f"❌ Error processing CSV data: {e}")
            return None
        
        # Determine confidence and prediction
        if avg_probability > 0.8 or avg_probability < 0.2:
            confidence = "High"
        elif avg_probability > 0.6 or avg_probability < 0.4:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        prediction = "EXOPLANET" if avg_probability > 0.5 else "NOT EXOPLANET"
        
        print(f"\n🎯 Prediction Results:")
        print(f"   Probability: {avg_probability:.4f} ({avg_probability*100:.2f}%)")
        print(f"   Prediction: {prediction}")
        print(f"   Confidence: {confidence}")
        
        return {
            'probability': avg_probability,
            'confidence': confidence,
            'prediction': prediction,
            'data_points': len(flux_clean) if 'flux_clean' in locals() else len(flux_resampled) if 'flux_resampled' in locals() else 0,
            'data_type': 'synthetic' if skip_to_prediction else 'real'
        }
    
    def show_feature_importance(self, top_n=10):
        """Show the most important features for exoplanet detection."""
        if self.model is None:
            print("❌ No model loaded")
            return
        
        print(f"\n🔍 Top {top_n} Most Important Features:")
        print("-" * 50)
        
        feature_importance = self.model.feature_importances_
        importance_indices = np.argsort(feature_importance)[::-1]
        
        for i in range(min(top_n, len(self.feature_names))):
            idx = importance_indices[i]
            print(f"{i+1:2d}. {self.feature_names[idx]:20s}: {feature_importance[idx]:.4f}")
    
    def analyze_prediction(self, light_curve):
        """Analyze a specific light curve and show feature contributions."""
        if self.model is None:
            print("❌ No model loaded")
            return None
        
        # Extract features using vectorized method
        features = extract_features_vectorized(np.array([light_curve]))
        features_scaled = self.scaler.transform(features)
        
        # Get prediction
        probability = self.model.predict_proba(features_scaled)[0, 1]
        
        # Get feature contributions (simplified)
        print(f"\n🔬 Feature Analysis:")
        print("-" * 40)
        print(f"Overall Probability: {probability:.4f} ({probability*100:.2f}%)")
        
        # Show top contributing features
        feature_importance = self.model.feature_importances_
        importance_indices = np.argsort(feature_importance)[::-1]
        
        print(f"\nTop Contributing Features:")
        for i in range(min(5, len(self.feature_names))):
            idx = importance_indices[i]
            feature_value = features[idx]
            importance = feature_importance[idx]
            print(f"  {self.feature_names[idx]:20s}: {feature_value:8.4f} (importance: {importance:.4f})")
        
        return probability

    def predict_from_fits_file(self, fits_path):
        """Load a .fits light curve file and make a prediction.

        Supports common Kepler/TESS light curve formats. Attempts to find TIME and
        a flux-like column (PDCSAP_FLUX, SAP_FLUX, FLUX). Normalizes and resamples
        to 2048 points before feature extraction.
        """
        try:
            result = self._load_flux_from_fits(fits_path)
            if result is None:
                return None
            flux_resampled, meta = result

            # Extract features and predict
            features = extract_features_vectorized(np.array([flux_resampled]))
            features_scaled = self.scaler.transform(features)
            probability = self.model.predict_proba(features_scaled)[0, 1]

            if probability > 0.8 or probability < 0.2:
                confidence = "High"
            elif probability > 0.6 or probability < 0.4:
                confidence = "Medium"
            else:
                confidence = "Low"

            prediction = "EXOPLANET" if probability > 0.5 else "NOT EXOPLANET"

            print(f"✅ FITS processed: {os.path.basename(fits_path)}")
            print(f"   Time column: {meta.get('time_col')}, Flux column: {meta.get('flux_col')}")
            print(f"   Data points (resampled): {len(flux_resampled)}")
            print(f"   Probability: {probability:.4f} ({probability*100:.2f}%)")
            print(f"   Prediction: {prediction}")
            print(f"   Confidence: {confidence}")

            return {
                'probability': probability,
                'confidence': confidence,
                'prediction': prediction,
                'data_points': len(flux_resampled),
                'data_type': 'fits'
            }
        except Exception as e:
            print(f"❌ Error processing FITS file: {e}")
            return None

    def _load_flux_from_fits(self, fits_path):
        """Helper: load and resample flux from a FITS file; returns (flux_2048, meta)."""
        try:
            try:
                from astropy.io import fits
            except Exception:
                print("❌ astropy is not installed. Install it with: pip install astropy")
                return None

            if not os.path.exists(fits_path):
                print(f"❌ File not found: {fits_path}")
                return None

            with fits.open(fits_path, memmap=False) as hdul:
                table_hdu = None
                for hdu in hdul:
                    if hasattr(hdu, 'data') and hdu.data is not None:
                        cols = [c.name.upper() for c in getattr(hdu, 'columns', [])]
                        if 'TIME' in cols:
                            table_hdu = hdu
                            break
                if table_hdu is None:
                    for hdu in hdul:
                        if hdu.__class__.__name__ == 'BinTableHDU' and hdu.data is not None:
                            table_hdu = hdu
                            break
                if table_hdu is None or table_hdu.data is None:
                    print("❌ No table HDU with light curve data found in FITS file")
                    return None

                data = table_hdu.data
                colnames = [name.upper() for name in data.columns.names]

                time_col_name = None
                for candidate in ['TIME', 'BJD', 'TMID', 'T', 'JD']:
                    if candidate in colnames:
                        time_col_name = candidate
                        break
                if time_col_name is None:
                    print(f"❌ Could not find a time column in FITS (available: {colnames})")
                    return None

                flux_col_name = None
                for candidate in ['PDCSAP_FLUX', 'SAP_FLUX', 'FLUX', 'DET_FLUX', 'LC_DETREND', 'NORMFLUX', 'SAP_FLUX_CORR']:
                    if candidate in colnames:
                        flux_col_name = candidate
                        break
                if flux_col_name is None:
                    for candidate in ['INTENSITY', 'COUNTS']:
                        if candidate in colnames:
                            flux_col_name = candidate
                            break
                if flux_col_name is None:
                    print(f"❌ Could not find a flux column in FITS (available: {colnames})")
                    return None

                time = np.array(data[time_col_name], dtype=float)
                flux = np.array(data[flux_col_name], dtype=float)

                valid_mask = ~(np.isnan(time) | np.isnan(flux) | np.isinf(time) | np.isinf(flux))
                time = time[valid_mask]
                flux = flux[valid_mask]

                if len(flux) < 100:
                    print(f"❌ Insufficient data points in FITS ({len(flux)}). Need at least 100 points.")
                    return None

                median_flux = np.median(flux)
                if median_flux != 0 and (median_flux > 10 or median_flux < 0.1):
                    flux = flux / median_flux
                flux = flux / np.median(flux)

                if len(flux) != 2048:
                    x_old = np.linspace(0, 1, len(flux))
                    x_new = np.linspace(0, 1, 2048)
                    f = interpolate.interp1d(x_old, flux, kind='linear', bounds_error=False, fill_value='extrapolate')
                    flux_resampled = f(x_new)
                else:
                    flux_resampled = flux

                meta = {'time_col': time_col_name, 'flux_col': flux_col_name}
                return flux_resampled, meta
        except Exception as e:
            print(f"❌ Error reading FITS file: {e}")
            return None

    def analyze_fits_with_gemini(self, fits_path):
        """Extract features from FITS and ask Gemini for an assessment with JSON output."""
        result = self._load_flux_from_fits(fits_path)
        if result is None:
            return None
        flux_resampled, meta = result

        # Build feature vector
        features = np.array(extract_features_vectorized(np.array([flux_resampled]))[0].tolist())
        formatted_arr_fstr = np.array([f"{x:.3e}" for x in features])

        # Prepare Gemini client
        api_key = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
        if not api_key:
            print("❌ Missing Gemini API key. Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable.")
            return None
        try:
            from google import genai
            client = genai.Client()
        except Exception as e:
            print(f"❌ Gemini SDK not available or failed to initialize: {e}")
            print("   Install with: pip install google-generativeai")
            return None

        # Strict JSON schema
        expected_schema = {
            "type": "object",
            "properties": {
                "assessment": {"type": "string"},
                "confidence": {"type": "string", "enum": ["Low", "Medium", "High"]},
                "reasoning": {"type": "string"},
                "notable_features": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["assessment", "confidence", "reasoning"],
            "additionalProperties": False
        }

        system_instructions = (
            "You are an astrophysics assistant analyzing exoplanet transit light curve features. "
            "Given a numeric feature vector extracted from a light curve, provide a concise assessment "
            "of whether the signal suggests an exoplanet transit, with a confidence level. Return strictly JSON only."
        )

        prompt = f"Is this a exoplanet based on this format: features = torch.stack([mean_vals, std_vals, min_vals, max_vals, median_vals, q10_vals, q25_vals, q75_vals, q90_vals, dips_1sigma.float(), dips_2sigma.float(), dips_3sigma.float(), peaks_1sigma.float(), peaks_2sigma.float(), transit_depth_1sigma, transit_depth_2sigma, transit_depth_min, variance_vals, rmse_vals, skewness, kurtosis, dominant_freqs, dominant_power, spectral_centroid, low_freq_power, mid_freq_power, high_freq_power, rolling_mean_mean, rolling_mean_std, rolling_mean_min, rolling_mean_max, trend_slopes, r_squared, autocorr_lag1, autocorr_lag5, autocorr_lag10, periodicity_strength], dim=1) {formatted_arr_fstr}"

        try:
            response = client.models.generate_content(
                model="gemini-2.5-pro",
                contents=prompt
            )
            print(response)
            return response
            # print(prompt)
            # return prompt
        except Exception as e:
            print(f"❌ Failed to parse Gemini response: {e}")
            return None

def main():
    """Main interactive function."""
    print("🔬 Interactive Exoplanet Prediction Tool")
    print("=" * 50)
    print("This tool uses Random Forest model with real TESS data for exoplanet predictions.")
    print("Features excellent sensitivity and interpretable results.")
    print()
    
    # Initialize predictor
    predictor = ExoplanetPredictor()
    
    while True:
        print("\n" + "=" * 50)
        print("Choose an option:")
        print("1. Paste CSV data from exoplanet archive")
        print("2. Get predictions from real TESS data")
        print("3. Show feature importance analysis")
        print("4. Analyze specific light curve")
        print("5. Load .fits light curve file")
        print("6. Analyze .fits with Gemini (JSON)")
        print("7. Exit")
        
        choice = input("\nEnter your choice (1-7): ").strip()
        
        if choice == '1':
            print("\n📋 Paste your CSV data below (press Ctrl+D or Ctrl+Z when done):")
            print("(Include headers in your CSV data)")
            print("-" * 50)
            
            # Read multi-line input
            lines = []
            try:
                while True:
                    line = input()
                    lines.append(line)
            except EOFError:
                pass
            
            csv_text = '\n'.join(lines)
            
            if csv_text.strip():
                # Parse CSV
                df = predictor.parse_csv_data(csv_text)
                if df is not None:
                    # Make predictions
                    result = predictor.predict_from_csv_data(df)
                    if result:
                        print(f"\n🎯 Final Result:")
                        print(f"   Probability: {result['probability']:.4f} ({result['probability']*100:.2f}%)")
                        print(f"   Prediction: {result['prediction']}")
                        print(f"   Confidence: {result['confidence']}")
                        
            else:
                print("❌ No CSV data provided")
        
        elif choice == '2':
            print("\n🔬 Getting predictions from real TESS data...")
            num_samples = input("Number of samples (default 5): ").strip()
            try:
                num_samples = int(num_samples) if num_samples else 5
            except ValueError:
                num_samples = 5
            
            predictions = predictor.predict_from_real_data(num_samples)
            if predictions:
                print(f"\n📊 Predictions from {len(predictions)} real TESS samples:")
                print("-" * 50)
                for i, pred in enumerate(predictions):
                    status = "✅" if pred['correct'] else "❌"
                    print(f"Sample {i+1}: {pred['probability']:.4f} ({pred['probability']*100:.2f}%) - {pred['prediction']} - {pred['confidence']} {status}")
                
                avg_prob = np.mean([p['probability'] for p in predictions])
                print(f"\nAverage probability: {avg_prob:.4f} ({avg_prob*100:.2f}%)")
        
        elif choice == '3':
            print("\n🔍 Feature Importance Analysis")
            predictor.show_feature_importance()
        
        elif choice == '4':
            print("\n🔬 Analyze Specific Light Curve")
            print("This will analyze a random real TESS light curve with detailed feature breakdown.")
            
            if predictor.real_data is not None:
                X_real, y_real = predictor.real_data
                # Select a random sample
                idx = np.random.randint(0, len(X_real))
                light_curve = X_real[idx]
                true_label = y_real[idx]
                
                print(f"\nAnalyzing TESS light curve sample {idx}:")
                print(f"True label: {'EXOPLANET' if true_label > 0.5 else 'NOT EXOPLANET'}")
                
                probability = predictor.analyze_prediction(light_curve)
                
                prediction = "EXOPLANET" if probability > 0.5 else "NOT EXOPLANET"
                correct = (probability > 0.5) == (true_label > 0.5)
                status = "✅ CORRECT" if correct else "❌ INCORRECT"
                
                print(f"\nFinal Prediction: {prediction} {status}")
            else:
                print("❌ No real data available")
        
        elif choice == '5':
            print("\n🗂️  Load .fits Light Curve File")
            fits_path = input("Enter path to .fits file: ").strip()
            result = predictor.predict_from_fits_file(fits_path)
            if result:
                print(f"\n🎯 Final Result:")
                print(f"   Probability: {result['probability']:.4f} ({result['probability']*100:.2f}%)")
                print(f"   Prediction: {result['prediction']}")
                print(f"   Confidence: {result['confidence']}")

        elif choice == '6':
            print("\n🤖 Analyze .fits with Gemini")
            fits_path = input("Enter path to .fits file: ").strip()
            _ = predictor.analyze_fits_with_gemini(fits_path)

        elif choice == '7':
            print("\n👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice. Please enter 1, 2, 3, 4, or 5.")

if __name__ == "__main__":
    main()
