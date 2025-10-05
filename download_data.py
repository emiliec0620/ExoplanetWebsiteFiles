import lightkurve as lk
import os
from astropy.time import Time
import warnings
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from astroquery.mast import Observations

# Suppress common, non-critical warnings from lightkurve
warnings.filterwarnings('ignore', category=lk.LightkurveWarning)

## ------------------- CONFIGURATION ------------------- ##
# Define the number of stars to download (one file per star)
NUM_STARS = 50  # Download data from 50 different stars (reduced for testing)

# Define the year for which you want to download data.
YEAR = 2023

# Define the directory where files will be saved.
DOWNLOAD_DIR = "tess_lightcurves"

# Choose download method: "exoplanet_hosts", "diverse_sectors", or "random_stars"
DOWNLOAD_METHOD = "diverse_sectors"  # Use mixed TIC IDs for better class balance

# Performance optimization settings
MAX_WORKERS = 8  # Number of parallel threads for downloading (reduced to avoid I/O conflicts)
BATCH_SIZE = 4  # Process stars in batches
CACHE_SEARCHES = True  # Cache search results to avoid repeated API calls
## ----------------------------------------------------- ##

# Global search cache to avoid repeated API calls
search_cache = {}
cache_lock = threading.Lock()

def cached_search_lightcurve(star_name, mission='TESS', author='SPOC', **kwargs):
    """
    Cached version of search_lightcurve to avoid repeated API calls.
    """
    if not CACHE_SEARCHES:
        return lk.search_lightcurve(star_name, mission=mission, author=author, **kwargs)
    
    cache_key = f"{star_name}_{mission}_{author}_{str(kwargs)}"
    
    with cache_lock:
        if cache_key in search_cache:
            print(f"📋 Using cached result for {star_name}")
            return search_cache[cache_key]
    
    # Perform the search
    result = lk.search_lightcurve(star_name, mission=mission, author=author, **kwargs)
    
    with cache_lock:
        search_cache[cache_key] = result
    
    return result

def download_single_star(star_name, download_dir, star_index, total_stars):
    """
    Download a single star's light curve data.
    """
    try:
        print(f"🔎 [{star_index}/{total_stars}] Searching for {star_name}...")
        
        # Use cached search
        search_result = cached_search_lightcurve(star_name, mission='TESS', author='SPOC')
        
        if len(search_result) == 0:
            print(f"❌ No TESS data found for {star_name}")
            return None
            
        manifest = Observations.download_products(
            search_result.table[0:1],
            download_dir=download_dir,
            mrp_only=False,
            verbose=False
        )
        
        if len(manifest) > 0:
            tic_id = search_result[0].target_name.split()[1]
            print(f"✅ Downloaded {star_name} (TIC {tic_id}) - file {star_index}/{total_stars}")
            return tic_id
        else:
            print(f"❌ No files downloaded for {star_name}")
            return None
        
    except Exception as e:
        print(f"⚠️ Error downloading {star_name}: {e}")
        return None

def download_diverse_tess_data(num_stars, year, download_dir):
    """
    Downloads TESS light curve files from diverse stars using known TIC IDs.
    Uses parallel processing and caching for faster downloads.

    Args:
        num_stars (int): Number of different stars to download data from.
        year (int): The calendar year to search for data.
        download_dir (str): The path to the directory to save the files.
    """
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
        print(f"✅ Created directory: '{download_dir}'")

    print(f"\n🔍 Searching for {num_stars} diverse stars with TESS data...")
    print(f"⚡ Using {MAX_WORKERS} parallel workers for faster downloads")
    
    # Mixed list of diverse TIC IDs (known exoplanet hosts + random stars)
    diverse_tic_ids = [
        # Known exoplanet hosts (from original list)
        "261136679", "150428135", "377780790",
        
        # New random TIC IDs (should provide more false positives)
        "50365310", "88863718", "124709665", "106997505", "238597883",
        "169904935", "156115721", "65212867", "440801822", "107782586",
        "231663901", "139853601", "114018671", "427508467", "97700520",
        "96246348", "155044736", "175310067", "182943944", "291555748",
        "341420329", "149603524", "140706664", "143994283", "47384844",
        "238374636", "141482386", "297967252", "388624270", "20318757",
        "447283466", "374908020", "336732616", "464296022", "304021498",
        "146589986", "149601557", "400595342", "361413119", "146172354",
        "363260203", "90544017", "461867584", "231670397", "310483807",
        "385624852", "360286627", "90448944", "463402815", "54044474",
        "309787037", "370745311", "384549882", "16288184", "144065872",
        "66818296", "259863352", "317060587", "112395568", "366989877",
        "320004517", "421894914", "323132914", "31553893", "380783252",
        "38846515", "101230735", "253990973", "299799658", "406976746",
        "79748331", "7088246", "201604954", "201642601", "6663331",
        "31858843", "92352620", "31858844", "120247528", "140830390",
        "158297421", "327301957", "351601843", "360742636", "369960846",
        "370133522", "379240073", "289793076", "161032923", "253728991",
        "261108236", "322270620", "380589029", "381979901", "468148930",
        "426077080", "343648136", "129319156", "29344935", "361034196",
        "193413306", "387079085", "382331352", "136274063", "375223080",
        "375225453", "360630575", "383390264", "290348383", "281459670",
        "318608749", "271581073", "311890977", "351603103", "451599528",
        "274151135", "384513078", "394561119", "295599256", "50309953",
        "355703913", "446549905", "412014494", "369864738", "290348382",
        "409934330", "379286801", "304100538", "295541511", "90919952",
        "261369656", "388104525", "143257766", "24094603", "304774444",
        "469782185", "128790976", "80166433", "405862830", "269450900",
        "425721385", "91576611", "97409519", "254113311", "198213332",
        "267489265"
    ]
    
    downloaded_stars = set()
    downloaded_count = 0
    
    # Process TIC IDs in parallel batches
    tic_batch = diverse_tic_ids[:num_stars]
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_tic = {
            executor.submit(
                download_single_tic_direct, 
                tic_id, 
                download_dir, 
                i + 1, 
                len(tic_batch)
            ): tic_id 
            for i, tic_id in enumerate(tic_batch)
        }
        
        for future in as_completed(future_to_tic):
            tic_id = future_to_tic[future]
            try:
                result = future.result()
                if result and tic_id not in downloaded_stars:
                    downloaded_stars.add(tic_id)
                    downloaded_count += 1
                    
            except Exception as e:
                print(f"⚠️ Error processing TIC {tic_id}: {e}")
    
    print(f"\n🎉 Successfully downloaded {downloaded_count} files from {len(downloaded_stars)} unique stars!")
    return downloaded_stars

def download_single_tic_direct(tic_id, download_dir, file_index, total_files):
    """
    Download a single TIC ID's light curve data directly.
    """
    try:
        print(f"🔎 [{file_index}/{total_files}] Searching for TIC {tic_id}...")
        
        # Use cached search with proper TIC format
        search_result = cached_search_lightcurve(f"TIC {tic_id}", mission='TESS', author='SPOC')

        if len(search_result) == 0:
            print(f"❌ No TESS data found for TIC {tic_id}")
            return None
            
        manifest = Observations.download_products(
            search_result.table[0:1],
            download_dir=download_dir,
            mrp_only=False,
            verbose=False
        )

        if len(manifest) > 0:
            print(f"✅ Downloaded TIC {tic_id} (file {file_index}/{total_files})")
            return tic_id
        else:
            print(f"❌ No files downloaded for TIC {tic_id}")
            return None
        
    except Exception as e:
        print(f"⚠️ Error downloading TIC {tic_id}: {e}")
        return None

def download_single_tic(tic_id, search_result, download_dir, file_index, total_files):
    """
    Download a single TIC ID's light curve data.
    """
    try:
        # Get the first observation for this TIC ID
        tic_search = search_result[search_result['target_name'] == tic_id]
        if len(tic_search) > 0:
            # Download only the first file for this star
            first_obs = tic_search[0]
            lc = first_obs.download(download_dir=download_dir)
            
            print(f"✅ Downloaded TIC {tic_id} (file {file_index}/{total_files})")
            return tic_id
            
    except Exception as e:
        print(f"⚠️ Error downloading TIC {tic_id}: {e}")
        return None

def download_exoplanet_host_stars(num_stars, download_dir):
    """
    Downloads TESS light curves from known exoplanet host stars (one file per star).
    Uses parallel processing for faster downloads.
    
    Args:
        num_stars (int): Number of different stars to download data from.
        download_dir (str): The path to the directory to save the files.
    """
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
        print(f"✅ Created directory: '{download_dir}'")

    print(f"\n🔍 Searching for {num_stars} exoplanet host stars with TESS data...")
    print(f"⚡ Using {MAX_WORKERS} parallel workers for faster downloads")
    
    # List of well-known exoplanet host stars
    known_exoplanet_stars = [
        "TRAPPIST-1", "Kepler-10", "TOI-700", "Proxima Cen", "HD 209458",
        "HD 189733", "WASP-12", "WASP-43", "HAT-P-7", "K2-18",
        "GJ 1214", "GJ 436", "GJ 3470", "GJ 1132", "LHS 1140",
        "K2-3", "K2-9", "K2-18", "K2-72", "K2-155",
        "TOI-125", "TOI-126", "TOI-127", "TOI-128", "TOI-129",
        "TOI-130", "TOI-131", "TOI-132", "TOI-133", "TOI-134",
        "TOI-135", "TOI-136", "TOI-137", "TOI-138", "TOI-139",
        "TOI-140", "TOI-141", "TOI-142", "TOI-143", "TOI-144",
        "TOI-145", "TOI-146", "TOI-147", "TOI-148", "TOI-149",
        "TOI-150", "TOI-151", "TOI-152", "TOI-153", "TOI-154",
        "HD 219134", "HD 40307", "HD 85512", "HD 97658", "HD 10180",
        "HD 40307", "HD 85512", "HD 97658", "HD 10180", "HD 20794",
        "HD 40307", "HD 85512", "HD 97658", "HD 10180", "HD 20794"
    ]
    
    # Limit to requested number of stars
    stars_to_process = known_exoplanet_stars[:num_stars]
    
    downloaded_stars = set()
    downloaded_count = 0
    
    # Process stars in parallel batches
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all download tasks
        future_to_star = {
            executor.submit(
                download_single_star, 
                star_name, 
                download_dir, 
                i+1, 
                len(stars_to_process)
            ): star_name 
            for i, star_name in enumerate(stars_to_process)
        }
        
        # Process completed downloads
        for future in as_completed(future_to_star):
            star_name = future_to_star[future]
            try:
                tic_id = future.result()
                if tic_id and tic_id not in downloaded_stars:
                    downloaded_stars.add(tic_id)
                    downloaded_count += 1
                    
                    if downloaded_count >= num_stars:
                        # Cancel remaining tasks if we have enough
                        for f in future_to_star:
                            f.cancel()
                        break
                        
            except Exception as e:
                print(f"⚠️ Error processing {star_name}: {e}")
    
    print(f"\n🎉 Successfully downloaded {downloaded_count} files from {len(downloaded_stars)} unique exoplanet host stars!")
    return downloaded_stars

def download_random_tess_stars(num_stars, download_dir):
    """
    Downloads TESS light curves from random TESS stars (not necessarily exoplanet hosts).
    This should provide better class balance with more false positives.
    
    Args:
        num_stars (int): Number of different stars to download data from.
        download_dir (str): The path to the directory to save the files.
    """
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
        print(f"✅ Created directory: '{download_dir}'")

    print(f"\n🔍 Searching for {num_stars} random TESS stars...")
    print(f"⚡ Using {MAX_WORKERS} parallel workers for faster downloads")
    
    # List of random TIC IDs (not necessarily exoplanet hosts)
    random_tic_ids = [
        "50365310", "88863718", "124709665", "106997505", "238597883",
        "169904935", "156115721", "65212867", "440801822", "107782586",
        "231663901", "139853601", "114018671", "427508467", "97700520",
        "96246348", "155044736", "175310067", "182943944", "291555748",
        "341420329", "149603524", "140706664", "143994283", "47384844",
        "238374636", "141482386", "297967252", "388624270", "20318757",
        "447283466", "374908020", "336732616", "464296022", "304021498",
        "146589986", "149601557", "400595342", "361413119", "146172354",
        "363260203", "90544017", "461867584", "231670397", "310483807",
        "385624852", "360286627", "90448944", "463402815", "54044474",
        "309787037", "370745311", "384549882", "16288184", "144065872",
        "66818296", "259863352", "317060587", "112395568", "366989877",
        "320004517", "421894914", "323132914", "31553893", "380783252",
        "38846515", "101230735", "253990973", "299799658", "406976746",
        "79748331", "7088246", "201604954", "201642601", "6663331",
        "31858843", "92352620", "31858844", "120247528", "140830390",
        "158297421", "327301957", "351601843", "360742636", "369960846",
        "370133522", "379240073", "289793076", "161032923", "253728991",
        "261108236", "322270620", "380589029", "381979901", "468148930",
        "426077080", "343648136", "129319156", "29344935", "361034196",
        "193413306", "387079085", "382331352", "136274063", "375223080",
        "375225453", "360630575", "383390264", "290348383", "281459670",
        "318608749", "271581073", "311890977", "351603103", "451599528",
        "274151135", "384513078", "394561119", "295599256", "50309953",
        "355703913", "446549905", "412014494", "369864738", "290348382",
        "409934330", "379286801", "304100538", "295541511", "90919952",
        "261369656", "388104525", "143257766", "24094603", "304774444",
        "469782185", "128790976", "80166433", "405862830", "269450900",
        "425721385", "91576611", "97409519", "254113311", "198213332",
        "267489265"
    ]
    
    downloaded_stars = set()
    downloaded_count = 0
    
    # Process TIC IDs in parallel batches
    tic_batch = random_tic_ids[:num_stars]
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_tic = {
            executor.submit(
                download_single_tic_direct, 
                tic_id, 
                download_dir, 
                i + 1, 
                len(tic_batch)
            ): tic_id 
            for i, tic_id in enumerate(tic_batch)
        }
        
        for future in as_completed(future_to_tic):
            tic_id = future_to_tic[future]
            try:
                result = future.result()
                if result and tic_id not in downloaded_stars:
                    downloaded_stars.add(tic_id)
                    downloaded_count += 1

            except Exception as e:
                print(f"⚠️ Error processing TIC {tic_id}: {e}")
    
    print(f"\n🎉 Successfully downloaded {downloaded_count} files from {len(downloaded_stars)} unique random TESS stars!")
    return downloaded_stars

# --- This block runs the main function when the script is executed ---
if __name__ == "__main__":
    start_time = time.time()
    
    print(f"🚀 Starting optimized TESS data download...")
    print(f"⚡ Performance optimizations enabled:")
    print(f"   - Parallel workers: {MAX_WORKERS}")
    print(f"   - Batch size: {BATCH_SIZE}")
    print(f"   - Search caching: {CACHE_SEARCHES}")
    
    if DOWNLOAD_METHOD == "exoplanet_hosts":
        # Use exoplanet host stars for better diversity and known dispositions
        downloaded_stars = download_exoplanet_host_stars(NUM_STARS, DOWNLOAD_DIR)
        method_name = "exoplanet host stars"
    elif DOWNLOAD_METHOD == "random_stars":
        # Use random TESS stars for better class balance
        downloaded_stars = download_random_tess_stars(NUM_STARS, DOWNLOAD_DIR)
        method_name = "random TESS stars"
    else:
        # Use diverse sectors approach
        downloaded_stars = download_diverse_tess_data(NUM_STARS, YEAR, DOWNLOAD_DIR)
        method_name = "diverse sectors"
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\n📊 Summary:")
    print(f"   - Method: {method_name}")
    print(f"   - Downloaded files: {len(downloaded_stars)}")
    print(f"   - Unique stars: {len(downloaded_stars)}")
    print(f"   - Directory: {DOWNLOAD_DIR}")
    print(f"   - Total time: {duration:.2f} seconds")
    try:
        print(f"   - Average time per star: {duration/len(downloaded_stars):.2f} seconds")
    except:
        print(f"   - Average time per star: N/A")
    print("\n🎉 Download process complete.")