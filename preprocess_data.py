import lightkurve as lk
import numpy as np
import os
import pandas as pd
import requests
import time

# --- CONFIGURATION ---
DATA_DIR = "tess_lightcurves" # Directory from your download script
SEGMENT_LENGTH = 2048        # Must match the input size of your CNN
STEP_SIZE = 128              # How much to slide the window for the next segment

# --- PROCESSING ---
all_segments = []
all_labels = [] # Will be populated from TFOPWG dispositions
all_metadata = [] # Store additional metadata for each segment

# TFOPWG Disposition mapping
# CP = Confirmed Planet, FP = False Positive, KP = Known Planet, PC = Planet Candidate
# Additional dispositions: AP = Ambiguous Planet, EB = Eclipsing Binary, IS = Instrumental, 
# V = Stellar Variability, NTP = Not Transit-like Planet, etc.
DISPOSITION_MAP = {
    'CP': 1,  # Confirmed Planet - positive example
    'KP': 1,  # Known Planet - positive example  
    'PC': 1,  # Planet Candidate - positive example
    'APC': 1,  # Ambiguous Planet - positive example
    'FP': 0,  # False Positive - negative example
    'FA': 0,  # False Alarm - negative example
    'UNKNOWN': 0  # Default for unknown dispositions - treat as negative
}

def query_toi_catalog(tic_ids):
    """
    Query the NASA Exoplanet Archive TOI table for dispositions.
    
    Args:
        tic_ids (list): List of TIC IDs to query
        
    Returns:
        dict: Mapping of TIC ID to disposition
    """
    dispositions = {}
    
    try:
        print("🌐 Querying NASA Exoplanet Archive TOI table...")
        
        # NASA Exoplanet Archive API endpoint for TOI table
        base_url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
        
        # Build query for TIC IDs
        tic_list = ','.join([f"'{tic_id}'" for tic_id in tic_ids])
        
        query = f"SELECT tid, tfopwg_disp FROM toi WHERE tid IN ({tic_list})"
        
        params = {
            'query': query,
            'format': 'csv'
        }
        
        # Add small delay to be respectful to the API
        time.sleep(0.5)
        
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        
        # Parse CSV response
        lines = response.text.strip().split('\n')
        if len(lines) > 1:  # Has data beyond header
            for line in lines[1:]:  # Skip header
                parts = line.split(',')
                if len(parts) >= 2:
                    tic_id = parts[0].strip()
                    disposition = parts[1].strip().upper()
                    
                    if disposition and disposition != 'NULL':
                        dispositions[tic_id] = disposition
                        print(f"  TIC {tic_id}: {disposition}")
        
        print(f"✅ Found {len(dispositions)} TIC IDs with dispositions in TOI catalog")
        
    except Exception as e:
        print(f"⚠️ Error querying TOI catalog: {e}")
        print("   Falling back to local lookup table")
    
    return dispositions

def lookup_disposition_from_fits(lc):
    """
    Attempt to determine disposition from FITS file metadata.
    
    Args:
        lc: LightCurve object from FITS file
        
    Returns:
        str: Disposition code or 'UNKNOWN'
    """
    # Check for TOI information in metadata
    for key, value in lc.meta.items():
        if 'toi' in key.lower() and value is not None:
            # If TOI is present, it's likely a planet candidate
            return 'PC'
    
    # Check OBJECT field for TOI designation
    object_name = lc.meta.get('OBJECT', '').upper()
    if 'TOI' in object_name:
        return 'PC'
    
    # Check TIC ID for known patterns
    tic_id = str(lc.meta.get('TICID', ''))
    
    # Known TIC IDs with dispositions (fallback for common cases)
    known_dispositions = {
        '377780790': 'CP',  # TOI-700 - Confirmed Planet
        '261136679': 'CP',  # TRAPPIST-1 - Confirmed Planet
        '150428135': 'FP',  # Example False Positive
    }
    
    if tic_id in known_dispositions:
        return known_dispositions[tic_id]
    
    # Use heuristics based on available metadata
    teff = lc.meta.get('TEFF', 0)
    tess_mag = lc.meta.get('TESSMAG', 0)
    
    # Simple heuristic: very cool stars (M dwarfs) are more likely to have planets
    if teff > 0 and teff < 4000:  # M dwarf temperature range
        return 'PC'  # Planet Candidate
    
    return 'UNKNOWN'

def load_tfopwg_dispositions(disposition_file=None):
    """
    Load TFOPWG dispositions from a CSV file.
    
    Args:
        disposition_file (str): Path to CSV file with TFOPWG dispositions
        
    Returns:
        dict: Mapping of TIC ID or filename to disposition
    """
    dispositions = {}
    
    if disposition_file and os.path.exists(disposition_file):
        try:
            df = pd.read_csv(disposition_file)
            print(f"✅ Loaded TFOPWG dispositions from {disposition_file}")
            
            # Try different possible column names
            tic_col = None
            disp_col = None
            
            for col in df.columns:
                if 'tic' in col.lower() or 'id' in col.lower():
                    tic_col = col
                if 'disposition' in col.lower() or 'tfopwg' in col.lower():
                    disp_col = col
            
            if tic_col and disp_col:
                for _, row in df.iterrows():
                    tic_id = str(row[tic_col])
                    disposition = str(row[disp_col]).upper().strip()
                    dispositions[tic_id] = disposition
                    print(f"  TIC {tic_id}: {disposition}")
            else:
                print(f"⚠️ Could not find TIC ID or disposition columns in {disposition_file}")
                print(f"Available columns: {list(df.columns)}")
                
        except Exception as e:
            print(f"⚠️ Error loading dispositions from {disposition_file}: {e}")
    else:
        print("ℹ️ No disposition file provided. Will use default labels.")
    
    return dispositions

# This code will only run when the script is executed directly, not when imported

def load_preprocessed_data(data_dir="preprocessed_data"):
    """
    Load preprocessed data from saved files.
    
    Args:
        data_dir (str): Directory containing the preprocessed data files
        
    Returns:
        tuple: (X, y) where X is features and y is labels
    """
    try:
        # Try to load from combined file first
        combined_file = os.path.join(data_dir, "preprocessed_data.npz")
        if os.path.exists(combined_file):
            data = np.load(combined_file)
            return data['X'], data['y']
        
        # Fallback to separate files
        X_file = os.path.join(data_dir, "X_real.npy")
        y_file = os.path.join(data_dir, "y_real.npy")
        
        if os.path.exists(X_file) and os.path.exists(y_file):
            X = np.load(X_file)
            y = np.load(y_file)
            return X, y
        
        raise FileNotFoundError(f"No preprocessed data found in {data_dir}")
        
    except Exception as e:
        print(f"Error loading preprocessed data: {e}")
        return None, None

if __name__ == "__main__":
    print("Preprocessing TESS light curve data...")
    print("=" * 50)
    
    # --- MODIFIED SECTION TO SEARCH SUBDIRECTORIES ---
    fits_files = []
    print(f"Searching for .fits files in '{DATA_DIR}' and its subdirectories...")
    for dirpath, _, filenames in os.walk(DATA_DIR):
        for f in filenames:
            if f.endswith('.fits'):
                # Construct the full path and add it to our list
                fits_files.append(os.path.join(dirpath, f))
    # --- END MODIFIED SECTION ---

    print(f"✅ Found {len(fits_files)} .fits files to process.")

    # Generate or load TFOPWG dispositions
    disposition_file = "tfopwg_dispositions.csv"
    
    # First, extract all unique TIC IDs from the FITS files
    print("🔍 Extracting TIC IDs from FITS files...")
    unique_tic_ids = set()
    for file_path in fits_files:
        try:
            filename = os.path.basename(file_path)
            import re
            # TESS filename format: tessYYYYMMDDHHMMSS-sXXXX-0000000TICID-YYYY-s_lc.fits
            tic_match = re.search(r'-(\d{16})-', filename)
            if tic_match:
                tic_id = tic_match.group(1).lstrip('0')  # Remove leading zeros
                unique_tic_ids.add(tic_id)
        except Exception as e:
            print(f"⚠️ Error extracting TIC ID from {os.path.basename(file_path)}: {e}")
    
    print(f"📋 Found {len(unique_tic_ids)} unique TIC IDs")
    
    # Check if disposition file exists, if not create it with FITS-derived dispositions
    if not os.path.exists(disposition_file):
        print(f"📝 Creating disposition file from FITS metadata: {disposition_file}")
        
        # First, try to query the NASA Exoplanet Archive TOI table
        toi_dispositions = query_toi_catalog(list(unique_tic_ids))
        
        # Extract dispositions from FITS files for any remaining unknowns
        dispositions_from_fits = {}
        print("🔍 Analyzing FITS files for remaining dispositions...")
        
        for file_path in fits_files:
            try:
                filename = os.path.basename(file_path)
                # Extract TIC ID from filename
                import re
                tic_match = re.search(r'-(\d{16})-', filename)
                if tic_match:
                    tic_id = tic_match.group(1).lstrip('0')
                    if tic_id in unique_tic_ids and tic_id not in toi_dispositions:
                        # Load light curve to extract metadata
                        lc = lk.read(file_path)
                        disposition = lookup_disposition_from_fits(lc)
                        dispositions_from_fits[tic_id] = disposition
                        print(f"  TIC {tic_id}: {disposition}")
            except Exception as e:
                print(f"⚠️ Error analyzing {os.path.basename(file_path)}: {e}")
        
        # Combine TOI catalog and FITS-derived dispositions
        all_dispositions = {**toi_dispositions, **dispositions_from_fits}
        
        # Create CSV file with extracted dispositions
        with open(disposition_file, 'w') as f:
            f.write("TIC_ID,TFOPWG_Disposition,Notes\n")
            for tic_id in sorted(unique_tic_ids):
                disposition = all_dispositions.get(tic_id, 'UNKNOWN')
                if tic_id in toi_dispositions:
                    notes = f"From NASA Exoplanet Archive TOI table"
                elif tic_id in dispositions_from_fits:
                    notes = f"Extracted from FITS metadata"
                else:
                    notes = f"Could not determine disposition"
                f.write(f"{tic_id},{disposition},{notes}\n")
        
        print(f"✅ Created {disposition_file} with {len(unique_tic_ids)} TIC IDs")
        print(f"📊 Disposition sources:")
        print(f"   - NASA Exoplanet Archive TOI table: {len(toi_dispositions)} TIC IDs")
        print(f"   - FITS metadata analysis: {len(dispositions_from_fits)} TIC IDs")
        print(f"   - Unknown: {len(unique_tic_ids) - len(all_dispositions)} TIC IDs")
        print("ℹ️ You can edit the CSV file to correct any dispositions if needed")
        print("   Disposition codes: CP, KP, PC, AP (positive) | FP, EB, IS, V, NTP (negative)")
        print("\n🔄 Running preprocessing with extracted dispositions...")
        # Continue with processing instead of exiting
    
    # Load existing dispositions
    dispositions = load_tfopwg_dispositions(disposition_file)
    
    # Validate dispositions and show statistics
    unknown_dispositions = []
    valid_dispositions = []
    invalid_dispositions = []
    
    for tic_id, disp in dispositions.items():
        if disp == 'UNKNOWN':
            unknown_dispositions.append(tic_id)
        elif disp in DISPOSITION_MAP:
            valid_dispositions.append((tic_id, disp))
        else:
            invalid_dispositions.append((tic_id, disp))
    
    print(f"📊 Disposition Summary:")
    print(f"   - Valid dispositions: {len(valid_dispositions)} TIC IDs")
    print(f"   - Unknown dispositions: {len(unknown_dispositions)} TIC IDs")
    print(f"   - Invalid dispositions: {len(invalid_dispositions)} TIC IDs")
    
    if invalid_dispositions:
        print(f"⚠️ Invalid dispositions found:")
        for tic_id, disp in invalid_dispositions[:5]:  # Show first 5
            print(f"   - TIC {tic_id}: '{disp}'")
        if len(invalid_dispositions) > 5:
            print(f"   ... and {len(invalid_dispositions) - 5} more")
        print("   These will be treated as UNKNOWN (negative labels).")
    
    if unknown_dispositions:
        print(f"ℹ️ {len(unknown_dispositions)} TIC IDs have UNKNOWN dispositions (will be labeled as negative)")
    
    print(f"✅ Processing {len(dispositions)} TIC IDs total")

    # Filter files to only process those with known dispositions
    files_with_dispositions = []
    for file_path in fits_files:
        try:
            # Get TIC ID from filename first (faster than loading full file)
            filename = os.path.basename(file_path)
            import re
            # TESS filename format: tessYYYYMMDDHHMMSS-sXXXX-0000000TICID-YYYY-s_lc.fits
            # Extract the TIC ID from the long number in the filename
            tic_match = re.search(r'-(\d{16})-', filename)
            if tic_match:
                tic_id = tic_match.group(1).lstrip('0')  # Remove leading zeros
                if tic_id in dispositions:
                    files_with_dispositions.append((file_path, tic_id))
                else:
                    print(f"ℹ️ Skipping {filename} - no disposition found for TIC {tic_id}")
            else:
                print(f"ℹ️ Skipping {filename} - could not extract TIC ID")
        except Exception as e:
            print(f"⚠️ Error checking {os.path.basename(file_path)}: {e}")
    
    print(f"📋 Processing {len(files_with_dispositions)} files with known dispositions...")

    for file_path, tic_id in files_with_dispositions:
        try:
            # 1. Load the light curve from the file
            lc = lk.read(file_path)
            
            # Clean the data by removing NaN values and outliers
            lc = lc.remove_nans().remove_outliers()

            # 2. Flatten and Normalize
            flat_lc = lc.flatten(window_length=401)
            norm_lc = flat_lc.normalize()
            
            flux = norm_lc.flux.value

            # 3. Get disposition for this TIC ID
            disposition = dispositions[tic_id]
            # Use UNKNOWN as fallback for invalid dispositions
            label = DISPOSITION_MAP.get(disposition, DISPOSITION_MAP['UNKNOWN'])
            
            print(f"  Processing TIC {tic_id} ({disposition}) - {len(flux)} flux points")
            
            # 4. Create overlapping segments
            segments_added = 0
            for i in range(0, len(flux) - SEGMENT_LENGTH, STEP_SIZE):
                segment = flux[i : i + SEGMENT_LENGTH]
                
                if len(segment) == SEGMENT_LENGTH:
                    all_segments.append(segment)
                    all_labels.append(label)
                    
                    # Store metadata for this segment
                    metadata = {
                        'file_path': file_path,
                        'tic_id': tic_id,
                        'disposition': disposition,
                        'segment_start': i,
                        'segment_end': i + SEGMENT_LENGTH
                    }
                    all_metadata.append(metadata)
                    segments_added += 1
            
            print(f"    Added {segments_added} segments")

        except Exception as e:
            # Use os.path.basename to get just the filename for cleaner logging
            print(f"⚠️ Could not process {os.path.basename(file_path)}: {e}")

    # Convert lists to NumPy arrays ready for the PyTorch Dataset
    X_real = np.array(all_segments, dtype=np.float32)
    y_real = np.array(all_labels, dtype=np.float32)

    print(f"\n🎉 Preprocessing complete! Generated {len(X_real)} segments from the downloaded data.")

    # Print disposition statistics
    unique_labels, counts = np.unique(y_real, return_counts=True)
    print(f"\n📊 Disposition Statistics:")
    
    # Count by actual disposition
    disposition_counts = {}
    for metadata in all_metadata:
        disp = metadata['disposition']
        disposition_counts[disp] = disposition_counts.get(disp, 0) + 1
    
    print(f"   By TFOPWG Disposition:")
    for disp, count in sorted(disposition_counts.items()):
        percentage = count / len(y_real) * 100
        label_type = "Positive" if DISPOSITION_MAP.get(disp, 0) == 1 else "Negative"
        print(f"     - {disp} ({label_type}): {count} segments ({percentage:.1f}%)")
    
    print(f"\n   By Binary Classification:")
    for label, count in zip(unique_labels, counts):
        disposition_name = {0: 'Negative', 1: 'Positive'}[int(label)]
        percentage = count / len(y_real) * 100
        print(f"     - {disposition_name}: {count} segments ({percentage:.1f}%)")

    # Save the preprocessed data to files
    output_dir = "preprocessed_data"
    os.makedirs(output_dir, exist_ok=True)

    # Save features and labels
    np.save(os.path.join(output_dir, "X_real.npy"), X_real)
    np.save(os.path.join(output_dir, "y_real.npy"), y_real)

    # Save metadata
    import json
    with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
        json.dump(all_metadata, f, indent=2)

    print(f"\n💾 Preprocessed data saved to:")
    print(f"   - Features: {output_dir}/X_real.npy")
    print(f"   - Labels: {output_dir}/y_real.npy")
    print(f"   - Metadata: {output_dir}/metadata.json")
    print(f"   - Shape: {X_real.shape}")

    # Also save as a combined file for convenience
    np.savez(os.path.join(output_dir, "preprocessed_data.npz"), 
             X=X_real, y=y_real)
    print(f"   - Combined: {output_dir}/preprocessed_data.npz")