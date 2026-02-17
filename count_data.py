import numpy as np
import glob
import os

data_dir = "./data"  # Change if your data is elsewhere
files = glob.glob(os.path.join(data_dir, "*.npz"))
total_images = 0

print(f"Scanning {len(files)} files...")

for f in files:
    try:
        with np.load(f) as data:
            if 'images' in data:
                n = len(data['images'])
                total_images += n
                print(f" - {os.path.basename(f)}: {n} images")
    except Exception as e:
        print(f"Error reading {f}: {e}")

print(f"\n>>> TOTAL DATASET SIZE: {total_images} frames")