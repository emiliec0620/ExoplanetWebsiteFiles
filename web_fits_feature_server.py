#!/usr/bin/env python3
"""
Simple Flask server to accept a .fits light curve upload and return
the resampled flux array (tensor-like data) and extracted features as JSON.
"""

import os
import io
import json
import numpy as np
from flask import Flask, request, jsonify, Response

# Reuse existing vectorized feature extractor
from random_forest_exoplanet_model import extract_features_vectorized
import pickle
import os

app = Flask(__name__)

# Load Random Forest model for predictions
RF_MODEL_PATH = "random_forest_exoplanet_model.pkl"
rf_model = None
rf_scaler = None

def load_rf_model():
    """Load the trained Random Forest model."""
    global rf_model, rf_scaler
    try:
        if os.path.exists(RF_MODEL_PATH):
            with open(RF_MODEL_PATH, 'rb') as f:
                model_data = pickle.load(f)
            rf_model = model_data['model']
            rf_scaler = model_data['scaler']
            print(f"✅ Random Forest model loaded from {RF_MODEL_PATH}")
        else:
            print(f"⚠️ Random Forest model not found at {RF_MODEL_PATH}")
    except Exception as e:
        print(f"❌ Error loading Random Forest model: {e}")

# Load model on startup
load_rf_model()


def load_flux_from_fits_bytes(file_bytes):
    """Load and resample flux from FITS bytes; returns (flux_2048, meta)."""
    try:
        try:
            from astropy.io import fits
        except Exception:
            return None, {"error": "astropy missing. Install with: pip install astropy"}

        with fits.open(io.BytesIO(file_bytes), memmap=False) as hdul:
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
                return None, {"error": "No table HDU with light curve data found in FITS file"}

            data = table_hdu.data
            colnames = [name.upper() for name in data.columns.names]

            time_col_name = None
            for candidate in ['TIME', 'BJD', 'TMID', 'T', 'JD']:
                if candidate in colnames:
                    time_col_name = candidate
                    break
            if time_col_name is None:
                return None, {"error": f"No time column found. Available: {colnames}"}

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
                return None, {"error": f"No flux column found. Available: {colnames}"}

            time = np.array(data[time_col_name], dtype=float)
            flux = np.array(data[flux_col_name], dtype=float)

            # Clean
            valid = ~(np.isnan(time) | np.isnan(flux) | np.isinf(time) | np.isinf(flux))
            time = time[valid]
            flux = flux[valid]
            if len(flux) < 100:
                return None, {"error": f"Insufficient data points: {len(flux)} (<100)"}

            # Normalize toward baseline 1.0
            median_flux = float(np.median(flux))
            if median_flux != 0 and (median_flux > 10 or median_flux < 0.1):
                flux = flux / median_flux
            flux = flux / float(np.median(flux))

            # Resample to 2048
            if len(flux) != 2048:
                from scipy import interpolate
                x_old = np.linspace(0, 1, len(flux))
                x_new = np.linspace(0, 1, 2048)
                f = interpolate.interp1d(x_old, flux, kind='linear', bounds_error=False, fill_value='extrapolate')
                flux_resampled = f(x_new)
            else:
                flux_resampled = flux

            meta = {"time_col": time_col_name, "flux_col": flux_col_name, "orig_len": int(len(flux))}
            return flux_resampled.astype(float), meta
    except Exception as e:
        return None, {"error": f"Exception: {e}"}


@app.get("/")
def index():
    html = (
        """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Exoplanet Detection Analyzer</title>
    <style>
      body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; padding: 24px; max-width: 900px; margin: auto; }
      .card { border: 1px solid #e5e7eb; border-radius: 12px; padding: 20px; }
      h1 { font-size: 1.5rem; margin: 0 0 8px; }
      p { color: #4b5563; }
      input[type=file] { margin: 12px 0; }
      button { background: #111827; color: white; border: none; padding: 10px 16px; border-radius: 8px; cursor: pointer; }
      pre { white-space: pre-wrap; word-break: break-word; background: #0b1021; color: #e5e7eb; padding: 12px; border-radius: 8px; }
    </style>
  </head>
  <body>
    <div class="card">
      <h1>Exoplanet Detection Analyzer</h1>
      <p>Upload a .fits light curve file to get AI-powered exoplanet detection analysis using Random Forest machine learning and Gemini AI.</p>
      <form id="upload-form">
        <input id="file" type="file" name="file" accept=".fits" required />
        <br />
        <button type="submit">Analyze for Exoplanets</button>
        <button id="open-gemini" type="button" disabled>Open in Gemini</button>
      </form>
      
      <div id="prediction-display" style="display: none; margin: 20px 0; padding: 20px; border-radius: 12px; text-align: center;">
        <h2 id="prediction-text" style="font-size: 2.5rem; margin: 0; font-weight: bold;"></h2>
        <p id="prediction-confidence" style="font-size: 1.2rem; margin: 10px 0 0; opacity: 0.8;"></p>
        <p id="prediction-probability" style="font-size: 1rem; margin: 5px 0 0; opacity: 0.6;"></p>
      </div>
      
      <h3>Analysis Results</h3>
      <pre id="output"></pre>
    </div>
    <script>
      const form = document.getElementById('upload-form');
      const output = document.getElementById('output');
      const openGeminiBtn = document.getElementById('open-gemini');
      const predictionDisplay = document.getElementById('prediction-display');
      const predictionText = document.getElementById('prediction-text');
      const predictionConfidence = document.getElementById('prediction-confidence');
      const predictionProbability = document.getElementById('prediction-probability');
      let lastGeminiPrompt = '';
      
      form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const fileInput = document.getElementById('file');
        if (!fileInput.files[0]) { alert('Select a .fits file'); return; }
        const fd = new FormData();
        fd.append('file', fileInput.files[0]);
        output.textContent = 'Uploading...';
        predictionDisplay.style.display = 'none';
        
        try {
          const res = await fetch('/api/extract', { method: 'POST', body: fd });
          const json = await res.json();
          
          // Display only the Gemini prompt and Random Forest prediction
          let displayContent = '';
          
          if (json.random_forest_prediction) {
            const pred = json.random_forest_prediction;
            displayContent += `🤖 Random Forest Prediction:\n`;
            displayContent += `Prediction: ${pred.prediction}\n`;
            displayContent += `Confidence: ${pred.confidence}\n`;
            displayContent += `Probability: ${(pred.probability * 100).toFixed(1)}%\n\n`;
            
            // Show Random Forest prediction prominently
            predictionText.textContent = pred.prediction;
            predictionConfidence.textContent = `Confidence: ${pred.confidence}`;
            predictionProbability.textContent = `Probability: ${(pred.probability * 100).toFixed(1)}%`;
            
            // Color coding
            if (pred.prediction === 'EXOPLANET') {
              predictionDisplay.style.backgroundColor = '#dcfce7';
              predictionDisplay.style.borderColor = '#16a34a';
              predictionText.style.color = '#16a34a';
            } else {
              predictionDisplay.style.backgroundColor = '#fef2f2';
              predictionDisplay.style.borderColor = '#dc2626';
              predictionText.style.color = '#dc2626';
            }
            predictionDisplay.style.display = 'block';
          }
          
          if (json.gemini_prompt_human) {
            displayContent += `📝 Gemini Prompt:\n${json.gemini_prompt_human}`;
            lastGeminiPrompt = json.gemini_prompt_human;
            openGeminiBtn.disabled = false;
          } else {
            openGeminiBtn.disabled = true;
          }
          
          output.textContent = displayContent;
          
        } catch (err) {
          output.textContent = 'Request failed: ' + err;
          openGeminiBtn.disabled = true;
          predictionDisplay.style.display = 'none';
        }
      });

      openGeminiBtn.addEventListener('click', () => {
        if (!lastGeminiPrompt) return;
        const encoded = encodeURIComponent(lastGeminiPrompt);
        // Example Gemini URL pattern that accepts a prompt param; adjust as needed
        const url = `https://gemini.google.com/app?prompt_text=${encoded}`;
        window.open(url, '_blank');
      });
    </script>
  </body>
</html>
        """
    )
    return Response(html, mimetype='text/html')


@app.post("/api/extract")
def api_extract():
    if 'file' not in request.files:
        return jsonify({"error": "No file field 'file' in form-data"}), 400
    f = request.files['file']
    if not f or f.filename == '':
        return jsonify({"error": "Empty file"}), 400
    if not f.filename.lower().endswith('.fits'):
        return jsonify({"error": "Only .fits files are supported"}), 400

    file_bytes = f.read()
    flux_resampled, meta = load_flux_from_fits_bytes(file_bytes)
    if flux_resampled is None:
        return jsonify({"error": meta.get('error', 'Failed to read FITS')}), 400

    # Extract features using existing vectorized function
    try:
        features = extract_features_vectorized(np.array([flux_resampled]))[0]
    except Exception as e:
        return jsonify({"error": f"Feature extraction failed: {e}"}), 500

    # Get Random Forest prediction if model is available
    rf_prediction = None
    if rf_model is not None and rf_scaler is not None:
        try:
            features_scaled = rf_scaler.transform([features])
            probability = rf_model.predict_proba(features_scaled)[0, 1]
            
            if probability > 0.8 or probability < 0.2:
                confidence = "High"
            elif probability > 0.6 or probability < 0.4:
                confidence = "Medium"
            else:
                confidence = "Low"
            
            prediction = "EXOPLANET" if probability > 0.5 else "NOT EXOPLANET"
            
            rf_prediction = {
                "probability": float(probability),
                "confidence": confidence,
                "prediction": prediction
            }
        except Exception as e:
            print(f"⚠️ Random Forest prediction failed: {e}")

    # Prepare JSON-safe outputs
    flux_list = flux_resampled.astype(float).tolist()
    features_list = np.array(features, dtype=float).tolist()
    features_sci = [f"{v:.3e}" for v in features_list]

    # Compose a ready-to-paste Gemini prompt (request human-readable output)
    feature_spec = (
        "Feature order corresponds to extract_features_vectorized: [mean, std, min, max, median, q10, q25, q75, q90, "
        "dips_1sigma, dips_2sigma, dips_3sigma, peaks_1sigma, peaks_2sigma, transit_depth_1sigma, transit_depth_2sigma, transit_depth_min, "
        "variance, rmse, skewness, kurtosis, dominant_freq, dominant_power, spectral_centroid, low_freq_power, mid_freq_power, high_freq_power, "
        "rolling_mean_mean, rolling_mean_std, rolling_mean_min, rolling_mean_max, trend_slope, r_squared, autocorr_lag1, autocorr_lag5, autocorr_lag10, periodicity_strength]."
    )
    gemini_prompt = (
        "You are an astrophysics assistant analyzing exoplanet transit light curve features. "
        + feature_spec + "\n"
        + "Given this feature vector (scientific notation):\nfeatures = " + json.dumps(features_sci) + "\n"
        + "Provide a brief, human-readable assessment of whether the signal suggests an exoplanet transit, include a confidence (Low/Medium/High), and a short rationale with notable features as bullet points."
    )

    # Human-readable prompt with labeled features
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
    labeled_lines = []
    for name, val in zip(feature_names, features_list):
        labeled_lines.append(f"- {name}: {val:.6g}")
    human_intro = (
        "Analyze the following light-curve feature summary and determine if it suggests an exoplanet transit. "
        "Provide a concise, human-readable explanation of the exoplanet with: a one-line assessment, a confidence (Low/Medium/High), and 3–5 bullet points highlighting notable features."
    )
    gemini_prompt_human = (
        human_intro + "\n\nDetected columns: time=" + meta.get('time_col', 'TIME') + ", flux=" + meta.get('flux_col', 'FLUX') +
        f"\nResampled length: {len(flux_resampled)}\n\nFeatures (labeled):\n" + "\n".join(labeled_lines)
    )

    response = {
        "gemini_prompt_human": gemini_prompt_human
    }
    
    # Add Random Forest prediction if available
    if rf_prediction is not None:
        response["random_forest_prediction"] = rf_prediction
    
    return jsonify(response)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', '5000'))
    app.run(host='0.0.0.0', port=port, debug=True)


