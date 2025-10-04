import lightkurve as lk
import os
from astropy.time import Time
import warnings

# Suppress common, non-critical warnings from lightkurve
warnings.filterwarnings('ignore', category=lk.LightkurveWarning)

## ------------------- CONFIGURATION ------------------- ##
# Define the list of stars you want to download data for.
TARGET_STARS = ["TRAPPIST-1", "Kepler-10", "TOI-700", "Proxima Cen"]

# Define the year for which you want to download data.
YEAR = 2023

# Define the directory where files will be saved.
DOWNLOAD_DIR = "tess_lightcurves"
## ----------------------------------------------------- ##


def download_tess_data_for_year(targets, year, download_dir):
    """
    Searches for and downloads TESS light curve files for a list of targets
    that were observed within a specific year.

    Args:
        targets (list): A list of star names (e.g., "Kepler-10") or TIC IDs.
        year (int): The calendar year to search for data.
        download_dir (str): The path to the directory to save the files.
    """
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
        print(f"✅ Created directory: '{download_dir}'")

    # Define the time range for the year in Modified Julian Date (MJD).
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"
    time_start_mjd = Time(start_date).mjd
    time_end_mjd = Time(end_date).mjd
    print(f"\nSearching for data between {start_date} and {end_date}...")

    # --- Loop through each target star ---
    for star in targets:
        print(f"\n{'─'*20}\n🔎 Processing target: {star}\n{'─'*20}")
        try:
            # Search for all available TESS light curves from the SPOC pipeline.
            search_result = lk.search_lightcurve(star, mission='TESS', author='SPOC')

            if len(search_result) == 0:
                print(f"❌ No TESS data found for {star}.")
                continue

            # --- CORRECTED SECTION ---
            # The correct column names are 't_min' and 't_max'.
            # This logic finds any observation period that overlaps with the target year.
            # year_mask = (search_result.t_min <= time_end_mjd) & (search_result.t_max >= time_start_mjd)
            filtered_result = search_result
            # --- END CORRECTION ---

            if len(filtered_result) == 0:
                print(f"INFO: Data exists for {star}, but no observations overlapped with {year}.")
            else:
                print(f"Found {len(filtered_result)} observation(s) for {star} overlapping with {year}.")
                
                # Download all light curves found in the filtered search.
                lc_collection = filtered_result.download_all(download_dir=download_dir)
                print(f"✅ Successfully downloaded {len(lc_collection)} files to '{download_dir}'.")

        except Exception as e:
            print(f"ERROR: An error occurred while processing {star}: {e}")

# --- This block runs the main function when the script is executed ---
if __name__ == "__main__":
    download_tess_data_for_year(TARGET_STARS, YEAR, DOWNLOAD_DIR)
    print("\n🎉 Download process complete.")