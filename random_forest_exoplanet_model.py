#!/usr/bin/env python3
"""
Random Forest-based exoplanet detection model for light curve time-series data.
This provides a baseline comparison with neural network approaches.
"""

import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from preprocess_data import load_preprocessed_data
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
import torch
import time

# For reproducibility
np.random.seed(42)

def check_mps_compatibility():
    """Check MPS compatibility and provide detailed information."""
    if torch.backends.mps.is_available():
        print("🍎 Apple Silicon MPS Support:")
        print(f"   MPS Available: {torch.backends.mps.is_available()}")
        print(f"   MPS Built: {torch.backends.mps.is_built()}")
        
        # Check PyTorch version compatibility
        torch_version = torch.__version__
        print(f"   PyTorch Version: {torch_version}")
        
        # Test basic MPS operations
        try:
            test_tensor = torch.randn(100, 100, device='mps')
            result = torch.matmul(test_tensor, test_tensor.T)
            print("   ✅ Basic MPS operations working")
            
            # Test FFT on MPS
            try:
                fft_test = torch.fft.fft(test_tensor)
                print("   ✅ MPS FFT operations working")
            except Exception as e:
                print(f"   ⚠️ MPS FFT limited: {e}")
                
        except Exception as e:
            print(f"   ❌ MPS operations failed: {e}")
            return False
        
        return True
    else:
        print("🍎 Apple Silicon MPS Support:")
        print("   MPS not available on this system")
        return False

def detect_device():
    """Detect the best available device for computation with enhanced MPS support."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"🚀 CUDA GPU detected: {torch.cuda.get_device_name()}")
        print(f"   CUDA version: {torch.version.cuda}")
        return device
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("🚀 Apple Silicon GPU (MPS) detected")
        print("   Using Metal Performance Shaders for acceleration")
        
        # MPS-specific optimizations
        try:
            # Test MPS functionality
            test_tensor = torch.randn(10, 10, device=device)
            test_result = torch.matmul(test_tensor, test_tensor.T)
            print("   ✅ MPS functionality verified")
        except Exception as e:
            print(f"   ⚠️ MPS test failed: {e}")
            print("   Falling back to CPU")
            device = torch.device('cpu')
        
        return device
    else:
        device = torch.device('cpu')
        print("💻 Using CPU for computation")
        return device

def extract_features_vectorized(light_curves, force_cpu=False):
    """
    Enhanced vectorized feature extraction for multiple light curves at once.
    Uses PyTorch for GPU acceleration when available (CUDA/MPS/CPU).
    
    Args:
        light_curves: Array of light curves to process
        force_cpu: If True, forces CPU-only computation for consistent performance
    """
    if force_cpu:
        device = torch.device('cpu')
        print("💻 Using CPU-only mode for consistent performance")
    else:
        device = detect_device()
    
    # Convert to PyTorch tensors with MPS-specific handling
    try:
        if isinstance(light_curves, np.ndarray):
            lc_tensor = torch.from_numpy(light_curves).float().to(device)
        else:
            lc_tensor = torch.stack([torch.from_numpy(lc).float() for lc in light_curves]).to(device)
    except Exception as e:
        if device.type == 'mps':
            print(f"⚠️ MPS tensor creation failed: {e}")
            print("   Falling back to CPU for tensor operations")
            device = torch.device('cpu')
            if isinstance(light_curves, np.ndarray):
                lc_tensor = torch.from_numpy(light_curves).float().to(device)
            else:
                lc_tensor = torch.stack([torch.from_numpy(lc).float() for lc in light_curves]).to(device)
        else:
            raise e
    
    batch_size, seq_len = lc_tensor.shape
    
    # Basic statistical features (vectorized)
    mean_vals = torch.mean(lc_tensor, dim=1)
    std_vals = torch.std(lc_tensor, dim=1)
    min_vals = torch.min(lc_tensor, dim=1)[0]
    max_vals = torch.max(lc_tensor, dim=1)[0]
    
    # Percentiles (optimized with efficient sorting-based approach)
    # Sort once and extract multiple percentiles for O(n log n) complexity
    sorted_lc, _ = torch.sort(lc_tensor, dim=1)
    n = sorted_lc.shape[1]
    
    # Calculate percentile indices efficiently
    q10_idx = int(0.10 * (n - 1))
    q25_idx = int(0.25 * (n - 1))
    median_idx = int(0.50 * (n - 1))
    q75_idx = int(0.75 * (n - 1))
    q90_idx = int(0.90 * (n - 1))
    
    q10_vals = sorted_lc[:, q10_idx]
    q25_vals = sorted_lc[:, q25_idx]
    median_vals = sorted_lc[:, median_idx]
    q75_vals = sorted_lc[:, q75_idx]
    q90_vals = sorted_lc[:, q90_idx]
    
    # Enhanced transit-related features (optimized)
    # Pre-compute thresholds to avoid repeated calculations
    mean_expanded = mean_vals.unsqueeze(1)
    std_expanded = std_vals.unsqueeze(1)
    
    # Multiple sigma thresholds for better transit detection
    dips_1sigma = torch.sum(lc_tensor < (mean_expanded - 1 * std_expanded), dim=1)
    dips_2sigma = torch.sum(lc_tensor < (mean_expanded - 2 * std_expanded), dim=1)
    dips_3sigma = torch.sum(lc_tensor < (mean_expanded - 3 * std_expanded), dim=1)
    
    peaks_1sigma = torch.sum(lc_tensor > (mean_expanded + 1 * std_expanded), dim=1)
    peaks_2sigma = torch.sum(lc_tensor > (mean_expanded + 2 * std_expanded), dim=1)
    
    # Transit depth calculations
    transit_depth_1sigma = (mean_vals - q10_vals) / mean_vals
    transit_depth_2sigma = (mean_vals - q25_vals) / mean_vals
    transit_depth_min = (mean_vals - min_vals) / mean_vals
    
    # Variability metrics (optimized)
    variance_vals = torch.var(lc_tensor, dim=1)
    rmse_vals = torch.sqrt(torch.mean((lc_tensor - mean_expanded)**2, dim=1))
    
    # Skewness and Kurtosis (higher moments) - optimized
    normalized_lc = (lc_tensor - mean_expanded) / std_expanded
    skewness = torch.mean(normalized_lc**3, dim=1)
    kurtosis = torch.mean(normalized_lc**4, dim=1)
    
    # FFT features (optimized for CPU performance)
    try:
        if force_cpu:
            # Use numpy FFT for better CPU performance
            fft_vals = np.fft.fft(light_curves, axis=1)
            fft_magnitude = np.abs(fft_vals)
        else:
            fft_vals = torch.fft.fft(lc_tensor, dim=1)
            fft_magnitude = torch.abs(fft_vals)
    except Exception as e:
        if device.type == 'mps':
            print(f"⚠️ MPS FFT failed: {e}")
            print("   Using CPU for FFT operations")
            # Move to CPU for FFT, then back to device
            lc_cpu = lc_tensor.cpu()
            fft_vals = torch.fft.fft(lc_cpu, dim=1)
            fft_magnitude = torch.abs(fft_vals)
            if device.type == 'mps':
                fft_magnitude = fft_magnitude.to(device)
        else:
            raise e
    
    # Convert to numpy for CPU-only mode
    if force_cpu:
        fft_magnitude = torch.from_numpy(fft_magnitude)
    
    # Dominant frequency and power (optimized)
    freqs = torch.fft.fftfreq(seq_len, device=device) if not force_cpu else torch.from_numpy(np.fft.fftfreq(seq_len))
    positive_freq_mask = freqs > 0
    positive_freqs = freqs[positive_freq_mask]
    positive_fft = fft_magnitude[:, positive_freq_mask]
    
    # Find dominant frequency for each sample (limit search to avoid unnecessary computation)
    search_limit = min(seq_len//4, 512)  # Limit search to first 512 frequencies for speed
    dominant_freq_indices = torch.argmax(positive_fft[:, :search_limit], dim=1)
    dominant_freqs = positive_freqs[dominant_freq_indices]
    dominant_power = positive_fft[torch.arange(batch_size), dominant_freq_indices]
    
    # Power in different frequency bands (optimized with fixed indices)
    band_size = min(seq_len//8, 256)  # Limit band sizes for speed
    low_freq_power = torch.sum(positive_fft[:, :band_size], dim=1)
    mid_freq_power = torch.sum(positive_fft[:, band_size:band_size*2], dim=1)
    high_freq_power = torch.sum(positive_fft[:, band_size*2:band_size*4], dim=1)
    
    # Spectral centroid (simplified for speed)
    spectral_centroid = torch.sum(positive_freqs.unsqueeze(0) * positive_fft, dim=1) / torch.sum(positive_fft, dim=1)
    
    # Rolling statistics (optimized with cumulative sum approach for O(n) complexity)
    window_size = min(50, seq_len // 4)
    if window_size > 1:
        try:
            # Use cumulative sum approach for O(n) rolling mean instead of convolution
            # Pad the beginning to handle edge effects
            padded_lc = torch.cat([lc_tensor[:, :1].repeat(1, window_size//2), lc_tensor], dim=1)
            
            # Compute cumulative sum
            cumsum = torch.cumsum(padded_lc, dim=1)
            
            # Compute rolling mean using difference of cumulative sums
            rolling_mean = (cumsum[:, window_size:] - cumsum[:, :-window_size]) / window_size
            
            # Ensure we have the right length by truncating if necessary
            if rolling_mean.shape[1] > seq_len:
                rolling_mean = rolling_mean[:, :seq_len]
            elif rolling_mean.shape[1] < seq_len:
                # Pad with mean values if shorter
                pad_size = seq_len - rolling_mean.shape[1]
                pad_values = mean_vals.unsqueeze(1).repeat(1, pad_size)
                rolling_mean = torch.cat([rolling_mean, pad_values], dim=1)
            
            rolling_mean_mean = torch.mean(rolling_mean, dim=1)
            rolling_mean_std = torch.std(rolling_mean, dim=1)
            rolling_mean_min = torch.min(rolling_mean, dim=1)[0]
            rolling_mean_max = torch.max(rolling_mean, dim=1)[0]
            
        except Exception as e:
            if device.type == 'mps':
                print(f"⚠️ MPS rolling statistics failed: {e}")
                print("   Using CPU for rolling statistics operations")
                # Fallback to CPU for rolling statistics
                lc_cpu = lc_tensor.cpu()
                mean_cpu = mean_vals.cpu()
                
                padded_lc_cpu = torch.cat([lc_cpu[:, :1].repeat(1, window_size//2), lc_cpu], dim=1)
                cumsum_cpu = torch.cumsum(padded_lc_cpu, dim=1)
                rolling_mean_cpu = (cumsum_cpu[:, window_size:] - cumsum_cpu[:, :-window_size]) / window_size
                
                if rolling_mean_cpu.shape[1] > seq_len:
                    rolling_mean_cpu = rolling_mean_cpu[:, :seq_len]
                elif rolling_mean_cpu.shape[1] < seq_len:
                    pad_size = seq_len - rolling_mean_cpu.shape[1]
                    pad_values_cpu = mean_cpu.unsqueeze(1).repeat(1, pad_size)
                    rolling_mean_cpu = torch.cat([rolling_mean_cpu, pad_values_cpu], dim=1)
                
                rolling_mean_mean = torch.mean(rolling_mean_cpu, dim=1)
                rolling_mean_std = torch.std(rolling_mean_cpu, dim=1)
                rolling_mean_min = torch.min(rolling_mean_cpu, dim=1)[0]
                rolling_mean_max = torch.max(rolling_mean_cpu, dim=1)[0]
                
                # Move results back to device if needed
                if device.type == 'mps':
                    rolling_mean_mean = rolling_mean_mean.to(device)
                    rolling_mean_std = rolling_mean_std.to(device)
                    rolling_mean_min = rolling_mean_min.to(device)
                    rolling_mean_max = rolling_mean_max.to(device)
            else:
                raise e
    else:
        rolling_mean_mean = mean_vals
        rolling_mean_std = std_vals
        rolling_mean_min = min_vals
        rolling_mean_max = max_vals
    
    # Trend analysis (optimized)
    x = torch.arange(seq_len, device=device).float()
    x_mean = torch.mean(x)
    x_centered = x - x_mean
    x_var = torch.sum(x_centered**2)
    
    # Pre-compute centered light curves
    lc_centered = lc_tensor - mean_expanded
    
    numerator = torch.sum(x_centered.unsqueeze(0) * lc_centered, dim=1)
    trend_slopes = numerator / x_var
    
    # R-squared for trend fit (simplified)
    y_pred = trend_slopes.unsqueeze(1) * x_centered.unsqueeze(0) + mean_expanded
    ss_res = torch.sum((lc_tensor - y_pred)**2, dim=1)
    ss_tot = torch.sum(lc_centered**2, dim=1)
    r_squared = 1 - (ss_res / ss_tot)
    
    # Autocorrelation features (multiple lags)
    autocorr_lag1 = torch.mean(lc_tensor[:, 1:] * lc_tensor[:, :-1], dim=1) / variance_vals
    autocorr_lag5 = torch.mean(lc_tensor[:, 5:] * lc_tensor[:, :-5], dim=1) / variance_vals
    autocorr_lag10 = torch.mean(lc_tensor[:, 10:] * lc_tensor[:, :-10], dim=1) / variance_vals
    
    # Periodicity detection (optimized with FFT-based autocorrelation)
    # Use FFT-based autocorrelation for O(n log n) complexity instead of O(n²)
    try:
        if force_cpu:
            # Use numpy for CPU-optimized autocorrelation
            fft_lc = np.fft.fft(light_curves, axis=1)
            autocorr_fft = np.fft.ifft(fft_lc * np.conj(fft_lc), axis=1).real
            
            # Normalize by variance
            autocorr_normalized = autocorr_fft / variance_vals.numpy().reshape(-1, 1)
            
            # Take only positive lags (first half)
            max_lag_idx = seq_len // 2
            autocorr_positive = autocorr_normalized[:, :max_lag_idx]
            
            # Find strongest periodic signal (skip lag 0 which is always 1.0)
            max_autocorr = np.max(autocorr_positive[:, 1:], axis=1)
            periodicity_strength = torch.from_numpy(max_autocorr)
        else:
            # Compute autocorrelation using FFT: autocorr = IFFT(FFT(x) * conj(FFT(x)))
            fft_lc = torch.fft.fft(lc_tensor, dim=1)
            autocorr_fft = torch.fft.ifft(fft_lc * torch.conj(fft_lc), dim=1).real
            
            # Normalize by variance
            autocorr_normalized = autocorr_fft / variance_vals.unsqueeze(1)
            
            # Take only positive lags (first half)
            max_lag_idx = seq_len // 2
            autocorr_positive = autocorr_normalized[:, :max_lag_idx]
            
            # Find strongest periodic signal (skip lag 0 which is always 1.0)
            max_autocorr, max_lag = torch.max(autocorr_positive[:, 1:], dim=1)
            periodicity_strength = max_autocorr
        
    except Exception as e:
        if device.type == 'mps':
            print(f"⚠️ MPS FFT autocorrelation failed: {e}")
            print("   Using CPU for FFT autocorrelation operations")
            # Fallback to CPU for FFT autocorrelation
            lc_cpu = lc_tensor.cpu()
            variance_cpu = variance_vals.cpu()
            
            fft_lc_cpu = torch.fft.fft(lc_cpu, dim=1)
            autocorr_fft_cpu = torch.fft.ifft(fft_lc_cpu * torch.conj(fft_lc_cpu), dim=1).real
            autocorr_normalized_cpu = autocorr_fft_cpu / variance_cpu.unsqueeze(1)
            
            max_lag_idx = seq_len // 2
            autocorr_positive_cpu = autocorr_normalized_cpu[:, :max_lag_idx]
            max_autocorr_cpu, max_lag_cpu = torch.max(autocorr_positive_cpu[:, 1:], dim=1)
            periodicity_strength = max_autocorr_cpu.to(device) if device.type == 'mps' else max_autocorr_cpu
        else:
            raise e
    
    # Combine all features
    features = torch.stack([
        mean_vals, std_vals, min_vals, max_vals, median_vals, 
        q10_vals, q25_vals, q75_vals, q90_vals,
        dips_1sigma.float(), dips_2sigma.float(), dips_3sigma.float(),
        peaks_1sigma.float(), peaks_2sigma.float(),
        transit_depth_1sigma, transit_depth_2sigma, transit_depth_min,
        variance_vals, rmse_vals, skewness, kurtosis,
        dominant_freqs, dominant_power, spectral_centroid,
        low_freq_power, mid_freq_power, high_freq_power,
        rolling_mean_mean, rolling_mean_std, rolling_mean_min, rolling_mean_max,
        trend_slopes, r_squared,
        autocorr_lag1, autocorr_lag5, autocorr_lag10, periodicity_strength
    ], dim=1)
    print(f"Features: {features}")
    
    return features.cpu().numpy()

def extract_features_from_light_curve(light_curve):
    """Extract statistical features from light curve for Random Forest."""
    features = []
    
    # Basic statistical features
    features.extend([
        np.mean(light_curve),
        np.std(light_curve),
        np.min(light_curve),
        np.max(light_curve),
        np.median(light_curve),
        np.percentile(light_curve, 25),
        np.percentile(light_curve, 75),
    ])
    
    # Transit-related features
    # Look for dips in the light curve
    flux_mean = np.mean(light_curve)
    flux_std = np.std(light_curve)
    
    # Count significant dips (below 2 sigma)
    dips = np.sum(light_curve < (flux_mean - 2 * flux_std))
    features.append(dips)
    
    # Count significant peaks (above 2 sigma)
    peaks = np.sum(light_curve > (flux_mean + 2 * flux_std))
    features.append(peaks)
    
    # Transit depth estimation
    min_flux = np.min(light_curve)
    transit_depth = (flux_mean - min_flux) / flux_mean
    features.append(transit_depth)
    
    # Variability metrics
    features.append(np.var(light_curve))
    features.append(np.sqrt(np.mean(np.square(light_curve - flux_mean))))
    
    # Spectral features (simplified)
    fft = np.fft.fft(light_curve)
    fft_magnitude = np.abs(fft)
    
    # Dominant frequency
    freqs = np.fft.fftfreq(len(light_curve))
    dominant_freq_idx = np.argmax(fft_magnitude[1:len(fft_magnitude)//2]) + 1
    dominant_freq = freqs[dominant_freq_idx]
    features.append(dominant_freq)
    
    # Power in different frequency bands
    low_freq_power = np.sum(fft_magnitude[1:len(fft_magnitude)//4])
    high_freq_power = np.sum(fft_magnitude[len(fft_magnitude)//4:len(fft_magnitude)//2])
    features.extend([low_freq_power, high_freq_power])
    
    # Rolling statistics
    window_size = 50
    rolling_mean = np.convolve(light_curve, np.ones(window_size)/window_size, mode='valid')
    rolling_std = np.array([np.std(light_curve[i:i+window_size]) for i in range(len(light_curve)-window_size+1)])
    
    features.extend([
        np.mean(rolling_mean),
        np.std(rolling_mean),
        np.mean(rolling_std),
        np.std(rolling_std)
    ])
    
    # Trend analysis
    x = np.arange(len(light_curve))
    trend_slope = np.polyfit(x, light_curve, 1)[0]
    features.append(trend_slope)
    
    # Autocorrelation features
    autocorr = np.correlate(light_curve, light_curve, mode='full')
    autocorr = autocorr[autocorr.size // 2:]
    autocorr = autocorr / autocorr[0]  # Normalize
    
    # Find first significant autocorrelation peak
    significant_lags = np.where(autocorr[1:] > 0.1)[0]
    if len(significant_lags) > 0:
        first_significant_lag = significant_lags[0] + 1
    else:
        first_significant_lag = 0
    features.append(first_significant_lag)
    
    return np.array(features)

def train_random_forest_model():
    """Train a Random Forest model on the preprocessed data."""
    print("🌲 Training Random Forest Model for Exoplanet Detection")
    print("=" * 60)
    
    # Check MPS compatibility first
    check_mps_compatibility()
    print()
    
    # Detect device for GPU acceleration
    device = detect_device()
    
    # Load data
    print("Loading preprocessed TESS data...")
    start_time = time.time()
    X_real, y_real = load_preprocessed_data("preprocessed_data")
    load_time = time.time() - start_time
    print(f"✅ Loaded preprocessed data: {len(X_real)} samples in {load_time:.2f}s")
    
    # Extract features from light curves using vectorized approach
    print("Extracting features from light curves using GPU acceleration...")
    feature_start_time = time.time()
    
    # Process in batches for memory efficiency (MPS-optimized)
    if device.type == 'mps':
        batch_size = 800  # MPS has different memory characteristics
        print(f"   Using MPS-optimized batch size: {batch_size}")
    elif device.type == 'cuda':
        batch_size = 1000
        print(f"   Using CUDA batch size: {batch_size}")
    else:
        batch_size = 500
        print(f"   Using CPU batch size: {batch_size}")
    X_features = []
    
    for i in range(0, len(X_real), batch_size):
        batch_end = min(i + batch_size, len(X_real))
        batch_lcs = X_real[i:batch_end]
        
        print(f"  Processing batch {i//batch_size + 1}/{(len(X_real) + batch_size - 1)//batch_size} "
              f"(samples {i}-{batch_end-1})")
        
        batch_features = extract_features_vectorized(batch_lcs)
        X_features.append(batch_features)
        
        # MPS memory management
        if device.type == 'mps':
            torch.mps.empty_cache()
    
    X_features = np.vstack(X_features)
    feature_time = time.time() - feature_start_time
    print(f"✅ Extracted features: {X_features.shape} in {feature_time:.2f}s")
    
    # Final MPS cleanup
    if device.type == 'mps':
        torch.mps.empty_cache()
        print("   🧹 MPS memory cache cleared")
    
    # Check class distribution
    positive_samples = np.sum(y_real)
    negative_samples = len(y_real) - positive_samples
    print(f"Class distribution: {positive_samples} positive, {negative_samples} negative")
    print(f"Positive ratio: {positive_samples/len(y_real):.3f}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y_real, test_size=0.2, random_state=42, stratify=y_real
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest
    print("\n🌲 Training Random Forest...")
    training_start_time = time.time()
    
    # Try to use GPU-accelerated Random Forest if available
    try:
        import cuml
        from cuml.ensemble import RandomForestClassifier as cuRFClassifier
        print("🚀 Using GPU-accelerated Random Forest (cuML)")
        
        rf_model = cuRFClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_streams=1  # cuML parameter
        )
        
        # Convert to cuML format
        X_train_gpu = cuml.DataFrame(X_train_scaled)
        y_train_gpu = cuml.Series(y_train)
        
        rf_model.fit(X_train_gpu, y_train_gpu)
        
        # Predictions
        X_test_gpu = cuml.DataFrame(X_test_scaled)
        y_pred = rf_model.predict(X_test_gpu).to_numpy()
        y_pred_proba = rf_model.predict_proba(X_test_gpu).to_numpy()[:, 1]
        
        print("✅ GPU Random Forest training completed")
        
    except ImportError:
        print("💻 Using CPU Random Forest (scikit-learn)")
        
        rf_model = RandomForestClassifier(
            n_estimators=200,  # More trees for better performance
            max_depth=50,     # Deeper trees (you already increased this)
            min_samples_split=2,  # Allow more splits
            min_samples_leaf=1,    # Allow smaller leaves
            max_features='sqrt',   # Use sqrt of features for each split
            random_state=42,
            n_jobs=-1,
            class_weight='balanced',  # Handle class imbalance
            bootstrap=True,
            oob_score=True,  # Enable out-of-bag scoring
            warm_start=False
        )
        
        rf_model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = rf_model.predict(X_test_scaled)
        y_pred_proba = rf_model.predict_proba(X_test_scaled)[:, 1]
    
    training_time = time.time() - training_start_time
    print(f"✅ Training completed in {training_time:.2f}s")
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    auc_score = roc_auc_score(y_test, y_pred_proba)
    print(f"AUC Score: {auc_score:.4f}")
    
    # Confidence analysis
    print(f"\n🎯 Confidence Analysis:")
    high_confidence_pos = np.sum((y_pred_proba > 0.8) & (y_test == 1))
    high_confidence_neg = np.sum((y_pred_proba < 0.2) & (y_test == 0))
    total_high_confidence = high_confidence_pos + high_confidence_neg
    print(f"High confidence predictions (>0.8 or <0.2): {total_high_confidence}/{len(y_test)} ({total_high_confidence/len(y_test)*100:.1f}%)")
    
    # Probability distribution analysis
    print(f"Probability range: {np.min(y_pred_proba):.3f} - {np.max(y_pred_proba):.3f}")
    print(f"Mean probability: {np.mean(y_pred_proba):.3f}")
    print(f"Std probability: {np.std(y_pred_proba):.3f}")
    
    # Out-of-bag score if available
    try:
        oob_score = rf_model.oob_score_
        print(f"Out-of-bag score: {oob_score:.4f}")
    except:
        pass
    
    # Feature importance
    feature_names = [
        'mean', 'std', 'min', 'max', 'median', 
        'q10', 'q25', 'q75', 'q90',
        'dips_1sigma', 'dips_2sigma', 'dips_3sigma',
        'peaks_1sigma', 'peaks_2sigma',
        'transit_depth_1sigma', 'transit_depth_2sigma', 'transit_depth_min',
        'variance', 'rmse', 'skewness', 'kurtosis',
        'dominant_freq', 'dominant_power', 'spectral_centroid',
        'low_freq_power', 'mid_freq_power', 'high_freq_power',
        'rolling_mean_mean', 'rolling_mean_std', 'rolling_mean_min', 'rolling_mean_max',
        'trend_slope', 'r_squared',
        'autocorr_lag1', 'autocorr_lag5', 'autocorr_lag10', 'periodicity_strength'
    ]
    
    # Handle feature importance for both cuML and scikit-learn
    try:
        feature_importance = rf_model.feature_importances_
    except AttributeError:
        # cuML might have different attribute name
        try:
            feature_importance = rf_model.feature_importances_.to_numpy()
        except:
            print("⚠️ Feature importance not available for this model type")
            feature_importance = np.ones(len(feature_names)) / len(feature_names)
    
    print("\n🔍 Top 10 Most Important Features:")
    importance_indices = np.argsort(feature_importance)[::-1]
    for i in range(min(10, len(feature_names))):
        idx = importance_indices[i]
        print(f"  {i+1:2d}. {feature_names[idx]:20s}: {feature_importance[idx]:.4f}")
    
    # Save model and scaler
    model_data = {
        'model': rf_model,
        'scaler': scaler,
        'feature_names': feature_names,
        'auc_score': auc_score
    }
    
    with open('random_forest_exoplanet_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\n💾 Random Forest model saved to 'random_forest_exoplanet_model.pkl'")
    print(f"📈 Model AUC Score: {auc_score:.4f}")
    
    # Performance summary
    total_time = time.time() - start_time
    print(f"\n⚡ Performance Summary:")
    print(f"   - Data loading: {load_time:.2f}s")
    print(f"   - Feature extraction: {feature_time:.2f}s")
    print(f"   - Model training: {training_time:.2f}s")
    print(f"   - Total time: {total_time:.2f}s")
    print(f"   - Device used: {device}")
    
    return rf_model, scaler, feature_names

def test_random_forest_sensitivity():
    """Test Random Forest model sensitivity to different inputs."""
    print("\n🧪 Testing Random Forest Model Sensitivity")
    print("=" * 50)
    
    # Load the trained model
    try:
        with open('random_forest_exoplanet_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        
        rf_model = model_data['model']
        scaler = model_data['scaler']
        feature_names = model_data['feature_names']
        print("✅ Random Forest model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading Random Forest model: {e}")
        print("Please train the model first by running this script.")
        return
    
    def create_test_pattern(pattern_type, base_flux=1.0, noise_level=0.01):
        """Create different test patterns for sensitivity testing."""
        pattern_length = 2048
        
        if pattern_type == "flat":
            # Flat signal (no transit)
            pattern = np.full(pattern_length, base_flux)
            
        elif pattern_type == "transit_shallow":
            # Shallow transit (realistic exoplanet) - more realistic shape
            pattern = np.full(pattern_length, base_flux)
            transit_start = pattern_length // 2 - 30
            transit_end = pattern_length // 2 + 30
            
            # Create more realistic transit shape with ingress/egress
            for i in range(transit_start, transit_end):
                if i < pattern_length // 2 - 10:  # Ingress
                    depth = 0.999 + 0.001 * (i - transit_start) / 20
                elif i > pattern_length // 2 + 10:  # Egress
                    depth = 0.999 + 0.001 * (transit_end - i) / 20
                else:  # Full transit
                    depth = 0.999
                pattern[i] *= depth
            
        elif pattern_type == "transit_deep":
            # Deep transit (potential false positive)
            pattern = np.full(pattern_length, base_flux)
            transit_start = pattern_length // 2 - 50
            transit_end = pattern_length // 2 + 50
            
            # Create realistic deep transit shape
            for i in range(transit_start, transit_end):
                if i < pattern_length // 2 - 15:  # Ingress
                    depth = 0.95 + 0.05 * (i - transit_start) / 35
                elif i > pattern_length // 2 + 15:  # Egress
                    depth = 0.95 + 0.05 * (transit_end - i) / 35
                else:  # Full transit
                    depth = 0.95
                pattern[i] *= depth
            
        elif pattern_type == "multiple_transits":
            # Multiple transits (periodic) - more realistic
            pattern = np.full(pattern_length, base_flux)
            period = 200
            transit_duration = 30
            
            for i in range(0, pattern_length, period):
                transit_start = i + period // 2 - transit_duration // 2
                transit_end = i + period // 2 + transit_duration // 2
                
                if transit_end < pattern_length:
                    # Create realistic transit shape
                    for j in range(transit_start, transit_end):
                        if j < i + period // 2 - 10:  # Ingress
                            depth = 0.998 + 0.002 * (j - transit_start) / 20
                        elif j > i + period // 2 + 10:  # Egress
                            depth = 0.998 + 0.002 * (transit_end - j) / 20
                        else:  # Full transit
                            depth = 0.998
                        pattern[j] *= depth
                    
        elif pattern_type == "noise_only":
            # Pure noise
            pattern = np.random.normal(base_flux, noise_level, pattern_length)
            
        elif pattern_type == "trend":
            # Linear trend
            pattern = np.linspace(base_flux - 0.1, base_flux + 0.1, pattern_length)
            
        elif pattern_type == "strong_transit":
            # Strong, clear transit signal
            pattern = np.full(pattern_length, base_flux)
            transit_start = pattern_length // 2 - 40
            transit_end = pattern_length // 2 + 40
            
            # Create very clear transit
            for i in range(transit_start, transit_end):
                if i < pattern_length // 2 - 15:  # Ingress
                    depth = 0.99 + 0.01 * (i - transit_start) / 25
                elif i > pattern_length // 2 + 15:  # Egress
                    depth = 0.99 + 0.01 * (transit_end - i) / 25
                else:  # Full transit
                    depth = 0.99
                pattern[i] *= depth
                
        else:
            # Random pattern
            pattern = np.random.normal(base_flux, noise_level, pattern_length)
        
        # Add noise
        noise = np.random.normal(0, noise_level, pattern_length)
        pattern = pattern + noise
        
        return pattern
    
    def predict_pattern(pattern):
        """Make prediction for a given pattern."""
        features = extract_features_vectorized(np.array([pattern]))
        features_scaled = scaler.transform(features)
        prob = rf_model.predict_proba(features_scaled)[0, 1]
        return prob
    
    # Test different patterns
    print("\n📊 Testing Different Light Curve Patterns:")
    
    patterns = {
        "Flat Signal (No Transit)": "flat",
        "Shallow Transit (Real Planet)": "transit_shallow", 
        "Deep Transit (False Positive)": "transit_deep",
        "Strong Clear Transit": "strong_transit",
        "Multiple Transits (Periodic)": "multiple_transits",
        "Pure Noise": "noise_only",
        "Linear Trend": "trend"
    }
    
    results = {}
    for name, pattern_type in patterns.items():
        pattern = create_test_pattern(pattern_type)
        prob = predict_pattern(pattern)
        results[name] = prob
        print(f"  {name:30s} -> Probability: {prob:.4f} ({prob*100:.2f}%)")
    
    # Test sensitivity to transit depth
    print("\n📊 Testing Sensitivity to Transit Depth:")
    depths = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05]  # 0% to 5%
    depth_results = {}
    for depth in depths:
        pattern = np.full(2048, 1.0)
        transit_start = 1024 - 50
        transit_end = 1024 + 50
        pattern[transit_start:transit_end] *= (1.0 - depth)
        
        # Add noise
        noise = np.random.normal(0, 0.01, 2048)
        pattern = pattern + noise
        
        prob = predict_pattern(pattern)
        depth_results[depth] = prob
        print(f"  Transit Depth: {depth*100:5.1f}% -> Probability: {prob:.4f} ({prob*100:.2f}%)")
    
    # Test sensitivity to noise level
    print("\n📊 Testing Sensitivity to Noise Level:")
    noise_levels = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
    noise_results = {}
    for noise_level in noise_levels:
        pattern = create_test_pattern("transit_shallow", noise_level=noise_level)
        prob = predict_pattern(pattern)
        noise_results[noise_level] = prob
        print(f"  Noise Level: {noise_level:5.3f} -> Probability: {prob:.4f} ({prob*100:.2f}%)")
    
    # Test with real data samples
    print("\n📊 Testing with Real Data Samples:")
    X_real, y_real = load_preprocessed_data("preprocessed_data")
    real_results = []
    for i in range(min(10, len(X_real))):
        prob = predict_pattern(X_real[i])
        real_results.append(prob)
        print(f"  Real Sample {i+1:2d} -> Probability: {prob:.4f} ({prob*100:.2f}%)")
    
    # Summary analysis
    print("\n" + "=" * 50)
    print("RANDOM FOREST SENSITIVITY ANALYSIS SUMMARY:")
    
    # Pattern variation
    pattern_probs = list(results.values())
    pattern_variation = max(pattern_probs) - min(pattern_probs)
    print(f"Pattern Variation: {pattern_variation:.4f}")
    
    # Depth sensitivity
    depth_probs = list(depth_results.values())
    depth_variation = max(depth_probs) - min(depth_probs)
    print(f"Depth Sensitivity: {depth_variation:.4f}")
    
    # Noise sensitivity
    noise_probs = list(noise_results.values())
    noise_variation = max(noise_probs) - min(noise_probs)
    print(f"Noise Sensitivity: {noise_variation:.4f}")
    
    # Real data variation
    if real_results:
        real_variation = max(real_results) - min(real_results)
        print(f"Real Data Variation: {real_variation:.4f}")
    
    # Overall assessment
    overall_variation = max(pattern_variation, depth_variation, noise_variation)
    print(f"\nOverall Sensitivity: {overall_variation:.4f}")
    
    if overall_variation > 0.3:
        print("✅ EXCELLENT sensitivity - model responds well to different inputs")
    elif overall_variation > 0.2:
        print("✅ GOOD sensitivity - model shows reasonable variation")
    elif overall_variation > 0.1:
        print("⚠️  MODERATE sensitivity - model shows some variation")
    else:
        print("❌ POOR sensitivity - model barely responds to different inputs")

if __name__ == "__main__":
    # Train the model first
    train_random_forest_model()
    
    # Then test sensitivity
    test_random_forest_sensitivity()
