import lightkurve as lk
import numpy as np
import os
import pandas as pd

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
DISPOSITION_MAP = {
    'CP': 1,  # Confirmed Planet - positive example
    'KP': 1,  # Known Planet - positive example  
    'PC': 1,  # Planet Candidate - positive example (treating as positive for training)
    'FP': 0,  # False Positive - negative example
    'UNKNOWN': 0  # Default for unknown dispositions
}

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

    # Load TFOPWG dispositions if available
    # You can provide a CSV file with columns: TIC_ID, TFOPWG_Disposition
    disposition_file = "tfopwg_dispositions.csv"  # Change this to your disposition file
    dispositions = load_tfopwg_dispositions(disposition_file)

    for file_path in fits_files:
        try:
            # 1. Load the light curve from the file
            lc = lk.read(file_path)
            
            # Clean the data by removing NaN values and outliers
            lc = lc.remove_nans().remove_outliers()

            # 2. Flatten and Normalize
            flat_lc = lc.flatten(window_length=401)
            norm_lc = flat_lc.normalize()
            
            flux = norm_lc.flux.value

            # 3. Get TIC ID from the light curve metadata
            tic_id = None
            for key in ['TICID', 'TIC_ID', 'TIC', 'TARGETID']:
                if key in lc.meta:
                    tic_id = str(lc.meta[key])
                    break
            
            # If TIC ID not found in metadata, try to extract from filename
            if tic_id is None:
                filename = os.path.basename(file_path)
                # Try to extract TIC ID from filename (common patterns)
                import re
                tic_match = re.search(r'tic(\d+)', filename, re.IGNORECASE)
                if tic_match:
                    tic_id = tic_match.group(1)
            
            # 4. Get disposition for this TIC ID
            disposition = dispositions.get(tic_id, 'UNKNOWN') if tic_id else 'UNKNOWN'
            label = DISPOSITION_MAP.get(disposition, 0)
            
            # 5. Create overlapping segments
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
    for label, count in zip(unique_labels, counts):
        disposition_name = {0: 'Negative (FP/Unknown)', 1: 'Positive (CP/KP/PC)'}[int(label)]
        percentage = count / len(y_real) * 100
        print(f"   - {disposition_name}: {count} segments ({percentage:.1f}%)")

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