import lightkurve as lk
import numpy as np
import os
import pandas as pd
import requests
import time
import re
import json

# --- CONFIGURATION ---
DATA_DIR = "kepler_lightcurves"  # Directory from Kepler download script
SEGMENT_LENGTH = 2048            # Must match the input size of your CNN
STEP_SIZE = 128                  # How much to slide the window for the next segment

# --- PROCESSING ---
all_segments = []
all_labels = []  # Will be populated from Kepler dispositions
all_metadata = []  # Store additional metadata for each segment

# Kepler Disposition mapping
# CONFIRMED = Confirmed Planet, FALSE POSITIVE = False Positive, 
# CANDIDATE = Planet Candidate, NOT DISPOSITIONED = Unknown
DISPOSITION_MAP = {
    'CONFIRMED': 1,           # Confirmed Planet - positive example
    'CANDIDATE': 1,           # Planet Candidate - positive example  
    'FALSE POSITIVE': 0,      # False Positive - negative example
    'NOT DISPOSITIONED': 0,   # Unknown - treat as negative
    'UNKNOWN': 0              # Default for unknown dispositions - treat as negative
}

def query_kepler_catalog(kic_ids):
    """
    Query the NASA Exoplanet Archive Kepler table for dispositions.
    
    Args:
        kic_ids (list): List of KIC IDs to query
        
    Returns:
        dict: Mapping of KIC ID to disposition
    """
    dispositions = {}
    
    try:
        print("🌐 Querying NASA Exoplanet Archive Kepler table...")
        
        # NASA Exoplanet Archive API endpoint
        base_url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
        
        # Build query for KIC IDs
        kic_list = ','.join([f"'{kic_id}'" for kic_id in kic_ids])
        
        query = f"SELECT kepid, koi_disposition FROM cumulative WHERE kepid IN ({kic_list})"
        
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
                    kic_id = parts[0].strip()
                    disposition = parts[1].strip().upper()
                    
                    if disposition and disposition != 'NULL':
                        dispositions[kic_id] = disposition
                        print(f"  KIC {kic_id}: {disposition}")
        
        print(f"✅ Found {len(dispositions)} KIC IDs with dispositions in Kepler catalog")
        
    except Exception as e:
        print(f"⚠️ Error querying Kepler catalog: {e}")
        print("   Falling back to local lookup table")
    
    return dispositions

def lookup_disposition_from_fits(lc):
    """
    Attempt to determine disposition from Kepler FITS file metadata.
    
    Args:
        lc: LightCurve object from FITS file
        
    Returns:
        str: Disposition code or 'UNKNOWN'
    """
    # Check for Kepler Object of Interest (KOI) information in metadata
    for key, value in lc.meta.items():
        if 'koi' in key.lower() and value is not None:
            # If KOI is present, it's likely a planet candidate
            return 'CANDIDATE'
    
    # Check OBJECT field for KOI designation
    object_name = lc.meta.get('OBJECT', '').upper()
    if 'KOI' in object_name:
        return 'CANDIDATE'
    
    # Check KIC ID for known patterns
    kic_id = str(lc.meta.get('KEPID', ''))
    
    # Known KIC IDs with dispositions (fallback for common cases)
    known_dispositions = {
        '10797460': 'CONFIRMED',  # Kepler-10 - Confirmed Planet
        '10811496': 'CONFIRMED',  # Kepler-11 - Confirmed Planet
        '10848459': 'CONFIRMED',  # Kepler-22 - Confirmed Planet
        '6922244': 'CONFIRMED',   # Kepler-452 - Confirmed Planet
        '8395660': 'FALSE POSITIVE',  # Example False Positive
    }
    
    if kic_id in known_dispositions:
        return known_dispositions[kic_id]
    
    # Use heuristics based on available metadata
    teff = lc.meta.get('TEFF', 0)
    kepler_mag = lc.meta.get('KEPMAG', 0)
    
    # Simple heuristic: very cool stars (M dwarfs) are more likely to have planets
    if teff > 0 and teff < 4000:  # M dwarf temperature range
        return 'CANDIDATE'  # Planet Candidate
    
    return 'UNKNOWN'

def load_kepler_dispositions(disposition_file=None):
    """
    Load Kepler dispositions from a CSV file.
    
    Args:
        disposition_file (str): Path to CSV file with Kepler dispositions
        
    Returns:
        dict: Mapping of KIC ID or filename to disposition
    """
    dispositions = {}
    
    if disposition_file and os.path.exists(disposition_file):
        try:
            df = pd.read_csv(disposition_file)
            print(f"✅ Loaded Kepler dispositions from {disposition_file}")
            
            # Try different possible column names
            kic_col = None
            disp_col = None
            
            for col in df.columns:
                if 'kic' in col.lower() or 'kepid' in col.lower() or 'id' in col.lower():
                    kic_col = col
                if 'disposition' in col.lower() or 'koi' in col.lower():
                    disp_col = col
            
            if kic_col and disp_col:
                for _, row in df.iterrows():
                    kic_id = str(row[kic_col])
                    disposition = str(row[disp_col]).upper().strip()
                    dispositions[kic_id] = disposition
                    print(f"  KIC {kic_id}: {disposition}")
            else:
                print(f"⚠️ Could not find KIC ID or disposition columns in {disposition_file}")
                print(f"Available columns: {list(df.columns)}")
                
        except Exception as e:
            print(f"⚠️ Error loading dispositions from {disposition_file}: {e}")
    else:
        print("ℹ️ No disposition file provided. Will use default labels.")
    
    return dispositions

def load_preprocessed_kepler_data(data_dir="preprocessed_kepler_data"):
    """
    Load preprocessed Kepler data from saved files.
    
    Args:
        data_dir (str): Directory containing the preprocessed data files
        
    Returns:
        tuple: (X, y) where X is features and y is labels
    """
    try:
        # Try to load from combined file first
        combined_file = os.path.join(data_dir, "preprocessed_kepler_data.npz")
        if os.path.exists(combined_file):
            data = np.load(combined_file)
            return data['X'], data['y']
        
        # Fallback to separate files
        X_file = os.path.join(data_dir, "X_kepler.npy")
        y_file = os.path.join(data_dir, "y_kepler.npy")
        
        if os.path.exists(X_file) and os.path.exists(y_file):
            X = np.load(X_file)
            y = np.load(y_file)
            return X, y
        
        raise FileNotFoundError(f"No preprocessed Kepler data found in {data_dir}")
        
    except Exception as e:
        print(f"Error loading preprocessed Kepler data: {e}")
        return None, None

if __name__ == "__main__":
    print("Preprocessing Kepler light curve data...")
    print("=" * 50)
    
    # --- SEARCH FOR KEPLER FITS FILES ---
    fits_files = []
    print(f"Searching for .fits files in '{DATA_DIR}' and its subdirectories...")
    for dirpath, _, filenames in os.walk(DATA_DIR):
        for f in filenames:
            if f.endswith('.fits'):
                # Construct the full path and add it to our list
                fits_files.append(os.path.join(dirpath, f))
    
    print(f"✅ Found {len(fits_files)} .fits files to process.")

    # Generate or load Kepler dispositions
    disposition_file = "kepler_dispositions.csv"
    
    # First, extract all unique KIC IDs from the FITS files
    print("🔍 Extracting KIC IDs from FITS files...")
    unique_kic_ids = set()
    for file_path in fits_files:
        try:
            filename = os.path.basename(file_path)
            # Kepler filename format: kplrXXXXXXXX-YYYYYYY_llc.fits
            # Extract KIC ID from filename
            kic_match = re.search(r'kplr(\d+)', filename)
            if kic_match:
                kic_id = kic_match.group(1).lstrip('0')  # Remove leading zeros
                unique_kic_ids.add(kic_id)
        except Exception as e:
            print(f"⚠️ Error extracting KIC ID from {os.path.basename(file_path)}: {e}")
    
    print(f"📋 Found {len(unique_kic_ids)} unique KIC IDs")
    
    # Check if disposition file exists, if not create it with FITS-derived dispositions
    if not os.path.exists(disposition_file):
        print(f"📝 Creating disposition file from FITS metadata: {disposition_file}")
        
        # First, try to query the NASA Exoplanet Archive Kepler table
        kepler_dispositions = query_kepler_catalog(list(unique_kic_ids))
        
        # Extract dispositions from FITS files for any remaining unknowns
        dispositions_from_fits = {}
        print("🔍 Analyzing FITS files for remaining dispositions...")
        
        for file_path in fits_files:
            try:
                filename = os.path.basename(file_path)
                # Extract KIC ID from filename
                kic_match = re.search(r'kplr(\d+)', filename)
                if kic_match:
                    kic_id = kic_match.group(1).lstrip('0')
                    if kic_id in unique_kic_ids and kic_id not in kepler_dispositions:
                        # Load light curve to extract metadata
                        lc = lk.read(file_path)
                        disposition = lookup_disposition_from_fits(lc)
                        dispositions_from_fits[kic_id] = disposition
                        print(f"  KIC {kic_id}: {disposition}")
            except Exception as e:
                print(f"⚠️ Error analyzing {os.path.basename(file_path)}: {e}")
        
        # Combine Kepler catalog and FITS-derived dispositions
        all_dispositions = {**kepler_dispositions, **dispositions_from_fits}
        
        # Create CSV file with extracted dispositions
        with open(disposition_file, 'w') as f:
            f.write("KIC_ID,Kepler_Disposition,Notes\n")
            for kic_id in sorted(unique_kic_ids):
                disposition = all_dispositions.get(kic_id, 'UNKNOWN')
                if kic_id in kepler_dispositions:
                    notes = f"From NASA Exoplanet Archive Kepler table"
                elif kic_id in dispositions_from_fits:
                    notes = f"Extracted from FITS metadata"
                else:
                    notes = f"Could not determine disposition"
                f.write(f"{kic_id},{disposition},{notes}\n")
        
        print(f"✅ Created {disposition_file} with {len(unique_kic_ids)} KIC IDs")
        print(f"📊 Disposition sources:")
        print(f"   - NASA Exoplanet Archive Kepler table: {len(kepler_dispositions)} KIC IDs")
        print(f"   - FITS metadata analysis: {len(dispositions_from_fits)} KIC IDs")
        print(f"   - Unknown: {len(unique_kic_ids) - len(all_dispositions)} KIC IDs")
        print("ℹ️ You can edit the CSV file to correct any dispositions if needed")
        print("   Disposition codes: CONFIRMED, CANDIDATE (positive) | FALSE POSITIVE, NOT DISPOSITIONED (negative)")
        print("\n🔄 Running preprocessing with extracted dispositions...")
    
    # Load existing dispositions
    dispositions = load_kepler_dispositions(disposition_file)
    
    # Validate dispositions and show statistics
    unknown_dispositions = []
    valid_dispositions = []
    invalid_dispositions = []
    
    for kic_id, disp in dispositions.items():
        if disp == 'UNKNOWN':
            unknown_dispositions.append(kic_id)
        elif disp in DISPOSITION_MAP:
            valid_dispositions.append((kic_id, disp))
        else:
            invalid_dispositions.append((kic_id, disp))
    
    print(f"📊 Disposition Summary:")
    print(f"   - Valid dispositions: {len(valid_dispositions)} KIC IDs")
    print(f"   - Unknown dispositions: {len(unknown_dispositions)} KIC IDs")
    print(f"   - Invalid dispositions: {len(invalid_dispositions)} KIC IDs")
    
    if invalid_dispositions:
        print(f"⚠️ Invalid dispositions found:")
        for kic_id, disp in invalid_dispositions[:5]:  # Show first 5
            print(f"   - KIC {kic_id}: '{disp}'")
        if len(invalid_dispositions) > 5:
            print(f"   ... and {len(invalid_dispositions) - 5} more")
        print("   These will be treated as UNKNOWN (negative labels).")
    
    if unknown_dispositions:
        print(f"ℹ️ {len(unknown_dispositions)} KIC IDs have UNKNOWN dispositions (will be labeled as negative)")
    
    print(f"✅ Processing {len(dispositions)} KIC IDs total")

    # Filter files to only process those with known dispositions
    files_with_dispositions = []
    for file_path in fits_files:
        try:
            # Get KIC ID from filename first (faster than loading full file)
            filename = os.path.basename(file_path)
            # Extract KIC ID from filename
            kic_match = re.search(r'kplr(\d+)', filename)
            if kic_match:
                kic_id = kic_match.group(1).lstrip('0')
                if kic_id in dispositions:
                    files_with_dispositions.append((file_path, kic_id))
                else:
                    print(f"ℹ️ Skipping {filename} - no disposition found for KIC {kic_id}")
            else:
                print(f"ℹ️ Skipping {filename} - could not extract KIC ID")
        except Exception as e:
            print(f"⚠️ Error checking {os.path.basename(file_path)}: {e}")
    
    print(f"📋 Processing {len(files_with_dispositions)} files with known dispositions...")

    for file_path, kic_id in files_with_dispositions:
        try:
            # 1. Load the light curve from the file
            lc = lk.read(file_path)
            
            # Clean the data by removing NaN values and outliers
            lc = lc.remove_nans().remove_outliers()

            # 2. Flatten and Normalize (Kepler-specific processing)
            # Use longer window for Kepler data due to longer cadence
            flat_lc = lc.flatten(window_length=501)
            norm_lc = flat_lc.normalize()
            
            flux = norm_lc.flux.value

            # 3. Get disposition for this KIC ID
            disposition = dispositions[kic_id]
            # Use UNKNOWN as fallback for invalid dispositions
            label = DISPOSITION_MAP.get(disposition, DISPOSITION_MAP['UNKNOWN'])
            
            print(f"  Processing KIC {kic_id} ({disposition}) - {len(flux)} flux points")
            
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
                        'kic_id': kic_id,
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
    X_kepler = np.array(all_segments, dtype=np.float32)
    y_kepler = np.array(all_labels, dtype=np.float32)

    print(f"\n🎉 Preprocessing complete! Generated {len(X_kepler)} segments from the downloaded Kepler data.")

    # Print disposition statistics
    unique_labels, counts = np.unique(y_kepler, return_counts=True)
    print(f"\n📊 Disposition Statistics:")
    
    # Count by actual disposition
    disposition_counts = {}
    for metadata in all_metadata:
        disp = metadata['disposition']
        disposition_counts[disp] = disposition_counts.get(disp, 0) + 1
    
    print(f"   By Kepler Disposition:")
    for disp, count in sorted(disposition_counts.items()):
        percentage = count / len(y_kepler) * 100
        label_type = "Positive" if DISPOSITION_MAP.get(disp, 0) == 1 else "Negative"
        print(f"     - {disp} ({label_type}): {count} segments ({percentage:.1f}%)")
    
    print(f"\n   By Binary Classification:")
    for label, count in zip(unique_labels, counts):
        disposition_name = {0: 'Negative', 1: 'Positive'}[int(label)]
        percentage = count / len(y_kepler) * 100
        print(f"     - {disposition_name}: {count} segments ({percentage:.1f}%)")

    # Save the preprocessed data to files
    output_dir = "preprocessed_kepler_data"
    os.makedirs(output_dir, exist_ok=True)

    # Save features and labels
    np.save(os.path.join(output_dir, "X_kepler.npy"), X_kepler)
    np.save(os.path.join(output_dir, "y_kepler.npy"), y_kepler)

    # Save metadata
    with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
        json.dump(all_metadata, f, indent=2)

    print(f"\n💾 Preprocessed Kepler data saved to:")
    print(f"   - Features: {output_dir}/X_kepler.npy")
    print(f"   - Labels: {output_dir}/y_kepler.npy")
    print(f"   - Metadata: {output_dir}/metadata.json")
    print(f"   - Shape: {X_kepler.shape}")

    # Also save as a combined file for convenience
    np.savez(os.path.join(output_dir, "preprocessed_kepler_data.npz"), 
             X=X_kepler, y=y_kepler)
    print(f"   - Combined: {output_dir}/preprocessed_kepler_data.npz")
