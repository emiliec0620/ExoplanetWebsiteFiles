import lightkurve as lk
import os
import warnings
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import argparse

from astroquery.mast import Observations

# Suppress common, non-critical warnings from lightkurve
warnings.filterwarnings('ignore', category=lk.LightkurveWarning)

## ------------------- CONFIGURATION ------------------- ##
# Define the number of stars to download (one file per star)
NUM_STARS_DEFAULT = 50

# Define the directory where files will be saved
DOWNLOAD_DIR_DEFAULT = "kepler_lightcurves"

# Choose download method: "diverse_sectors" (Kepler analogue: diverse quarters)
DOWNLOAD_METHOD = "diverse_sectors"

# Performance optimization settings
MAX_WORKERS = 8  # Number of parallel threads for downloading
BATCH_SIZE = 4   # Process stars in batches
CACHE_SEARCHES = True  # Cache search results to avoid repeated API calls
## ----------------------------------------------------- ##

# Global search cache to avoid repeated API calls
search_cache = {}
cache_lock = threading.Lock()


def cached_search_lightcurve(target_name, mission='Kepler', **kwargs):
    """
    Cached version of search_lightcurve to avoid repeated API calls.
    """
    if not CACHE_SEARCHES:
        return lk.search_lightcurve(target_name, mission=mission, **kwargs)

    cache_key = f"{target_name}_{mission}_{str(kwargs)}"

    with cache_lock:
        if cache_key in search_cache:
            print(f"📋 Using cached result for {target_name}")
            return search_cache[cache_key]

    # Perform the search (first without author, then fall back to generic if needed)
    result = lk.search_lightcurve(target_name, mission=mission, **kwargs)

    with cache_lock:
        search_cache[cache_key] = result

    return result


def download_single_kic_direct(kic_id, download_dir, file_index, total_files):
    """
    Download a single KIC ID's light curve data directly (Kepler mission).
    """
    try:
        print(f"🔎 [{file_index}/{total_files}] Searching for KIC {kic_id}...")

        # Use cached search with proper KIC format
        search_result = cached_search_lightcurve(f"KIC {kic_id}", mission='Kepler')

        if len(search_result) == 0:
            print(f"❌ No Kepler data found for KIC {kic_id}")
            return None

        manifest = Observations.download_products(
            search_result.table[0:1],
            download_dir=download_dir,
            mrp_only=False,
            verbose=False
        )

        if len(manifest) > 0:
            print(f"✅ Downloaded KIC {kic_id} (file {file_index}/{total_files})")
            return kic_id
        else:
            print(f"❌ No files downloaded for KIC {kic_id}")
            return None

    except Exception as e:
        print(f"⚠️ Error downloading KIC {kic_id}: {e}")
        return None


def download_diverse_kepler_data(num_stars, download_dir):
    """
    Downloads Kepler light curve files from a diverse list of KIC IDs
    (analogue of the TESS "diverse_sectors" flow).

    Args:
        num_stars (int): Number of different stars to download data from.
        download_dir (str): The path to the directory to save the files.
    """
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
        print(f"✅ Created directory: '{download_dir}'")

    print(f"\n🔍 Searching for {num_stars} diverse Kepler stars (KIC IDs)...")
    print(f"⚡ Using {MAX_WORKERS} parallel workers for faster downloads")

    # Provided KIC IDs (deduplicated, order preserved)
    provided_kic_ids_raw = [
        10797460, 10797460, 10811496, 10848459, 10854555, 10872983, 10872983, 10872983, 6721123, 10910878,
        11446443, 10666592, 6922244, 10984090, 10419211, 10464078, 10480982, 10485250, 10526549, 10583066,
        10583180, 10601284, 10601284, 10601284, 2306756, 10662202, 10682541, 11460018, 11463211, 11465813,
        11493732, 11507101, 10910878, 10910878, 10987985, 11018648, 11018648, 11138155, 11152159, 11153539,
        11242721, 9579641, 11304958, 11391957, 11403044, 11414511, 11442465, 12110942, 12366084, 12366084,
        12404086, 8395660, 9579641, 11656840, 11754553, 11754553, 11754553, 11812062, 11818800, 11853255,
        11909839, 11918099, 11918099, 9579641, 11923270, 11960862, 12020329, 12066335, 12066335, 12070811,
        8395660, 3120431, 3246984, 3342970, 3342970, 12459725, 12470844, 12470844, 12644822, 2440757,
        2445129, 2713049, 3114167, 3114661, 3115833, 4139816, 4139816, 8395660, 4139816, 4275191, 4476123,
        3351888, 3453214, 3554600, 3641726, 3734868, 3832474, 8395660, 3832474, 3832474, 3836375, 3838486,
        3935914, 3940418, 4049131, 4139816, 5287983, 5358241, 10875245, 5358241, 5358241, 5358624, 4544670,
        4664847, 4725681, 4725681, 4913852, 4932348, 4936180, 11913073, 10875245, 5021899, 5077629, 5115978,
        5164255, 5252423, 5252423, 5272878, 5283542, 5649215, 5651104, 5792202, 10875245, 5792202, 5370302,
        5372966, 5376067, 5436502, 5436502, 5436502, 5436502, 10875245, 5436502, 5456651, 5456651, 5481416,
        5531576, 5531576, 5534814, 6291653, 6392727, 6422070, 6428700, 6428700, 5792202, 5794379, 5794379,
        5881688, 6022556, 6032497, 6061119, 6191521, 6267425, 3531558, 6276477, 6863998, 6867155, 6948054,
        6948054, 6948054, 6435936, 6522242, 6526710, 6587280, 9471974, 6587280, 6599919, 6675056, 6685526,
        6756669, 6784235, 6849310, 6849310, 6849310, 6862328, 9471974, 6862603, 7366258, 7366258, 7366258,
        7366258, 7373451, 6948054, 6949607, 6949607, 7031517, 11869052, 7109675, 7109675, 7118364, 7134976,
        7134976, 7135852, 7270230, 7287995, 7287995, 7287995, 3247396, 7303253, 5094751, 7678434, 7678434,
        7685981, 7708215, 7373451, 7377033, 7380537, 7434875, 8349582, 7434875, 7434875, 7455287, 7455287,
        7455287, 7458762, 7552344, 757450, 7585481, 7663691, 5812701, 8013419, 8018547, 8039892, 8150320,
        8150320, 11086270, 7767559, 7825899, 7825899, 7849854, 7870390, 7870390, 5094751, 7870390, 7907423,
        7907423, 7907423, 7938496, 8490993, 8490993, 8505670, 8505670, 8544996, 8552202, 8150320, 8150320,
        8150320, 8180063, 8226994, 8226994, 8226994, 8247638, 8247638, 8247638, 11086270, 8247638, 8255887,
        8256049, 8414716, 9140402, 9141746, 9159275, 9166862,
    ]

    seen = set()
    kic_ids = []
    for kid in provided_kic_ids_raw:
        if kid not in seen:
            seen.add(kid)
            kic_ids.append(str(kid))

    downloaded_stars = set()
    downloaded_count = 0

    # Process KIC IDs in parallel batches
    kic_batch = kic_ids[:num_stars]

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_kic = {
            executor.submit(
                download_single_kic_direct,
                kic_id,
                download_dir,
                i + 1,
                len(kic_batch)
            ): kic_id
            for i, kic_id in enumerate(kic_batch)
        }

        for future in as_completed(future_to_kic):
            kic_id = future_to_kic[future]
            try:
                result = future.result()
                if result and kic_id not in downloaded_stars:
                    downloaded_stars.add(kic_id)
                    downloaded_count += 1
            except Exception as e:
                print(f"⚠️ Error processing KIC {kic_id}: {e}")

    print(f"\n🎉 Successfully downloaded {downloaded_count} files from {len(downloaded_stars)} unique Kepler stars!")
    return downloaded_stars


def main():
    parser = argparse.ArgumentParser(description="Download Kepler light curves for provided KIC IDs (diverse sectors analogue)")
    parser.add_argument("--num", type=int, default=NUM_STARS_DEFAULT, help="Number of distinct KIC IDs to download")
    parser.add_argument("--download-dir", type=str, default=DOWNLOAD_DIR_DEFAULT, help="Directory to store downloads")
    args = parser.parse_args()

    start_time = time.time()

    print(f"🚀 Starting optimized Kepler data download...")
    print(f"⚡ Performance optimizations enabled:")
    print(f"   - Parallel workers: {MAX_WORKERS}")
    print(f"   - Batch size: {BATCH_SIZE}")
    print(f"   - Search caching: {CACHE_SEARCHES}")

    if DOWNLOAD_METHOD == "diverse_sectors":
        downloaded_stars = download_diverse_kepler_data(args.num, args.download_dir)
        method_name = "diverse sectors (Kepler analog)"
    else:
        downloaded_stars = download_diverse_kepler_data(args.num, args.download_dir)
        method_name = "diverse sectors (Kepler analog)"

    end_time = time.time()
    duration = end_time - start_time

    print(f"\n📊 Summary:")
    print(f"   - Method: {method_name}")
    print(f"   - Downloaded files: {len(downloaded_stars)}")
    print(f"   - Unique stars: {len(downloaded_stars)}")
    print(f"   - Directory: {args.download_dir}")
    print(f"   - Total time: {duration:.2f} seconds")
    try:
        print(f"   - Average time per star: {duration/max(1, len(downloaded_stars)):.2f} seconds")
    except Exception:
        print(f"   - Average time per star: N/A")
    print("\n🎉 Download process complete.")


if __name__ == "__main__":
    main()


