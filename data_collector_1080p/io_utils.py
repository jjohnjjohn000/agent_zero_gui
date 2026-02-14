# data_collector/io_utils.py
import os
import sys
import time
import numpy as np
from datetime import datetime

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
    
from config import DATA_DIR, COMPRESSION_LEVEL

def ensure_data_dir():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
def save_chunk(frames, mouse_data, keyboard_data, meta_data=None):
    """
    Saves a chunk of recording data to disk.
    frames: List of numpy arrays (images)
    mouse_data: List of dicts
    keyboard_data: List of dicts
    """
    ensure_data_dir()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"session_{timestamp}.npz"
    filepath = os.path.join(DATA_DIR, filename)

    frames_np = np.stack(frames)
    
    # Save the monitor index data too
    np.savez_compressed(
        filepath,
        images=frames_np,
        mouse=np.array(mouse_data, dtype=object),
        keyboard=np.array(keyboard_data, dtype=object),
        meta=np.array(meta_data, dtype=object) if meta_data else np.array([])
    )
    
    print(f"[Storage] Saved chunk: {filename} | {len(frames)} frames")