"""
=============================================================================
TITAN ENGINE v5.2 - SPATIAL VISION TRAINING SYSTEM (COMPLETE)
=============================================================================
Architecture: Spatial VAE (PixelShuffle / ResBlock / GroupNorm)
Optimization: Threaded I/O, RAM Panic Switches, Gradient Accumulation, AMP
Features: Auto-Recovery, Sanity Check, Heartbeat, GUI Integration
=============================================================================
"""

import os
import sys
import shutil
import gc
import glob
import json
import signal
import tempfile
import uuid
import atexit
import time
import logging
import traceback
import threading
import queue
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

# --- CRITICAL: ENVIRONMENT SETUP ---
# Prevents library clashes and optimizes AMD/NVIDIA allocation
os.environ['MIOPEN_DISABLE_CACHE'] = '1'
os.environ['PYTORCH_HIP_ALLOC_CONF'] = 'max_split_size_mb:128'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

# --- IMPORTS ---
try:
    import psutil
    import numpy as np
    import torch
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from torchvision.utils import save_image
except ImportError:
    pass

# Import Local Modules
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_dir
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Try importing the new Spatial VAE
try:
    from vae import VAE, vae_loss_function
except ImportError:
    try:
        from models.vae import VAE, vae_loss_function
    except:
        print("CRITICAL: model/vae.py not found.")

# =============================================================================
# --- CONFIGURATION & TUNING ---
# =============================================================================
DEFAULT_CONFIG = {
    # --- OPTIMIZER ---
    "LEARNING_RATE": 1e-4,     # Standard for AdamW + PixelShuffle
    "BATCH_SIZE": 16,           # Tight fit for 12GB VRAM
    "GRAD_ACCUM_STEPS": 4,     # Virtual Batch Size = 16
    "EPOCHS": 500,
    
    # --- ARCHITECTURE ---
    "LATENT_DIM": 32,          # Spatial Grid Channels (32x32x32)
    "CROP_SIZE": 256,          # Input Size
    
    # --- PERFORMANCE ---
    "NUM_THREADS": 6,          # Worker Threads
    "MAX_IMAGES_PER_POOL": 1500, # Reduced slightly for safety
    "MIN_SYS_RAM_GB": 4.0,     # Safety Cutoff
    "FILES_PER_LOAD": 10,
    
    # --- WEIGHTS (THE RECIPE) ---
    "MSE_WEIGHT": 50.0,        # Pixel Accuracy (L1)
    "PERCEPTUAL_WEIGHT": 2.0,  # VGG Structure
    "BETA": 0.0001,            # KL Divergence
    
    # --- IO ---
    "VISUALIZE_EVERY_N_POOLS": 5,
    "CHECKPOINT_EVERY_N_POOLS": 2,
    "DATA_DIR": os.path.join(project_root, "data"),
    "CHECKPOINT_DIR": os.path.join(project_root, "checkpoints"),
    "RESULTS_DIR": os.path.join(project_root, "results")
}

logger = logging.getLogger("TITAN")

# =============================================================================
# --- SYSTEM SAFETY MONITOR ---
# =============================================================================
def cleanup_memory():
    """Aggressive Garage Collection"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def check_panic_switches(min_sys_ram):
    """
    Checks System RAM and VRAM. 
    Returns True if we are about to crash the OS.
    """
    # 1. Check System RAM
    mem = psutil.virtual_memory()
    avail_sys_gb = mem.available / (1024**3)
    
    if avail_sys_gb < min_sys_ram:
        logger.critical(f"PANIC: System RAM Low ({avail_sys_gb:.2f} GB). Pausing.")
        return True
    
    # 2. Check VRAM (Safety margin for Desktop Window Manager)
    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / (1024**3)
        if peak > 11.5:
            logger.warning(f"VRAM WARNING: Peak usage {peak:.2f} GB. Flushing.")
            cleanup_memory()
            
    return False

atexit.register(cleanup_memory)

# =============================================================================
# --- DATA LOADING ENGINE (THREADED & ROBUST) ---
# =============================================================================
def load_single_file(args):
    """
    Worker function to load a single NPZ file.
    Handles cropping high-res screenshots down to 256x256.
    """
    fp, crop_size = args
    try:
        with np.load(fp) as data:
            if 'images' not in data: return None
            
            raw_images = data['images'] # Shape: (N, H, W, 3)
            n, h, w, c = raw_images.shape
            
            # --- CASE 1: Perfect Match ---
            if h == crop_size and w == crop_size:
                return raw_images.copy()

            # --- CASE 2: High Res (Need Crop) ---
            elif h > crop_size and w > crop_size:
                # Random Spatial Crop (The "Scanning" effect)
                y = np.random.randint(0, h - crop_size, size=n)
                x = np.random.randint(0, w - crop_size, size=n)
                
                patches = np.zeros((n, crop_size, crop_size, 3), dtype=np.uint8)
                for i in range(n):
                    patches[i] = raw_images[i, y[i]:y[i]+crop_size, x[i]:x[i]+crop_size]
                
                return patches # Assumes RGB
            
            return None
    except Exception as e:
        return None

def background_loader_manager(all_files, start_ptr, pool_queue, stop_event, cfg):
    """
    The 'Feeder' Thread. 
    Keeps the GPU fed so it never idles.
    """
    current_ptr = start_ptr
    total_files = len(all_files)
    
    # ThreadPool for Disk I/O
    with ThreadPoolExecutor(max_workers=cfg["NUM_THREADS"]) as executor:
        while not stop_event.is_set():
            vram_pool = []
            pool_size = 0
            
            # --- POOL BUILDING LOOP ---
            while pool_size < cfg["MAX_IMAGES_PER_POOL"] and not stop_event.is_set():
                end_ptr = min(current_ptr + cfg["FILES_PER_LOAD"], total_files)
                files_to_open = all_files[current_ptr : end_ptr]
                
                if not files_to_open:
                    current_ptr = 0 # Loop the dataset
                    continue
                
                # Launch Workers
                args = [(f, cfg["CROP_SIZE"]) for f in files_to_open]
                results = list(executor.map(load_single_file, args))
                
                for res in results:
                    if res is not None:
                        # Pre-Normalize to 0-1 range and Float32 here to save Main Thread time
                        tensor = torch.from_numpy(res).float().permute(0, 3, 1, 2) / 255.0
                        vram_pool.append(tensor)
                        pool_size += res.shape[0]
                
                current_ptr = end_ptr

            # --- PUSH TO MAIN QUEUE ---
            if vram_pool and not stop_event.is_set():
                try:
                    # Cat into one giant tensor for this pool
                    full_pool = torch.cat(vram_pool, dim=0)
                    pool_queue.put((full_pool, current_ptr), timeout=5)
                except queue.Full:
                    pass
            
            # Breathe
            time.sleep(0.1)

# =============================================================================
# --- STATE MANAGEMENT (SAVE/LOAD) ---
# =============================================================================
def save_titan_checkpoint(model, optimizer, epoch, file_ptr, loss, cfg):
    """Saves atomic checkpoint + JSON metadata"""
    weights_path = os.path.join(cfg["CHECKPOINT_DIR"], "vae_spatial.pth")
    state_path = os.path.join(cfg["CHECKPOINT_DIR"], "training_state.json")
    
    # 1. Weights
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, weights_path)

    # 2. Meta
    state = {
        "timestamp": datetime.now().isoformat(),
        "epoch": epoch, 
        "file_index": file_ptr, 
        "last_loss": float(loss),
        "lr": optimizer.param_groups[0]['lr']
    }
    with open(state_path, 'w') as f:
        json.dump(state, f, indent=4)
    
    logger.info(f" >>> SAVED CHECKPOINT: Epoch {epoch} | Loss {loss:.4f}")

def load_titan_checkpoint(model, optimizer, cfg):
    """Smart Restoration"""
    weights_path = os.path.join(cfg["CHECKPOINT_DIR"], "vae_spatial.pth")
    state_path = os.path.join(cfg["CHECKPOINT_DIR"], "training_state.json")
    
    start_epoch = 0
    start_file_ptr = 0
    
    if os.path.exists(weights_path):
        try:
            checkpoint = torch.load(weights_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info(" >>> MODEL WEIGHTS RESTORED.")
        except Exception as e:
            logger.error(f" >>> WEIGHT LOAD FAILED (Architecture Mismatch?): {e}")
            return 0, 0 # Start fresh if arch changed

    if os.path.exists(state_path):
        try:
            with open(state_path, 'r') as f:
                data = json.load(f)
                start_epoch = data.get("epoch", 0)
                start_file_ptr = data.get("file_index", 0)
                logger.info(f" >>> STATE RESTORED: Epoch {start_epoch}")
        except: pass
            
    return start_epoch, start_file_ptr

# =============================================================================
# --- THE MAIN PROCESS (THE BEAST) ---
# =============================================================================
class QueueLogger(logging.Handler):
    """Redirects logs to the GUI Queue"""
    def __init__(self, queue):
        super().__init__()
        self.queue = queue
    def emit(self, record):
        try:
            self.queue.put({"status": "LOG", "msg": self.format(record)})
        except: pass

def train_process_entrypoint(status_queue, stop_event, config_overrides):
    """
    Main Training Loop.
    Runs inside a separate process to protect the GUI from hanging.
    """
    # 1. SETUP
    cfg = DEFAULT_CONFIG.copy()
    if config_overrides: cfg.update(config_overrides)
    
    os.makedirs(cfg["CHECKPOINT_DIR"], exist_ok=True)
    os.makedirs(cfg["RESULTS_DIR"], exist_ok=True)
    
    # Logging Setup
    log_fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    logger.setLevel(logging.INFO)
    logger.handlers = []
    
    # File Handler
    fh = logging.FileHandler(os.path.join(cfg["CHECKPOINT_DIR"], "titan.log"))
    fh.setFormatter(log_fmt)
    logger.addHandler(fh)
    
    # GUI Handler
    gh = QueueLogger(status_queue)
    gh.setFormatter(log_fmt)
    logger.addHandler(gh)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"TITAN SPATIAL ENGINE v5.2 STARTED ON {device}")
    
    # 2. MODEL INITIALIZATION
    try:
        model = VAE().to(device)
        logger.info("Spatial VAE (PixelShuffle) Initialized.")
    except Exception as e:
        logger.critical(f"Model Init Failed: {e}")
        status_queue.put({"status": "ERROR", "msg": str(e)})
        return

    # Optimizer (AdamW is better for Transformers/Spatial than Adam)
    optimizer = optim.AdamW(model.parameters(), lr=cfg["LEARNING_RATE"], weight_decay=1e-5)
    
    # Mixed Precision Scaler
    scaler = torch.amp.GradScaler('cuda')
    
    # Load Checkpoint
    start_epoch, start_file_ptr = load_titan_checkpoint(model, optimizer, cfg)
    
    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )
    
    # =========================================================================
    # --- SANITY CHECK (PRE-FLIGHT) ---
    # =========================================================================
    logger.info(">>> RUNNING SANITY CHECK...")
    try:
        dummy = torch.randn(2, 3, 256, 256).to(device)
        with torch.amp.autocast('cuda'):
            r, m, l = model(dummy)
            loss_check = vae_loss_function(r, dummy, m, l, 0.0, 1.0, 0, 0, 0.0)
        scaler.scale(loss_check).backward()
        optimizer.zero_grad(set_to_none=True)
        del dummy, r, m, l, loss_check
        logger.info(">>> SANITY CHECK PASSED. ENGINE GREEN.")
    except Exception as e:
        logger.critical(f"SANITY CHECK FAILED: {e}")
        status_queue.put({"status": "ERROR", "msg": f"GPU/Model Fail: {e}"})
        return

    # 3. DATA PIPELINE INIT
    data_pattern = os.path.join(cfg["DATA_DIR"], "*.npz")
    all_files = sorted(glob.glob(data_pattern))
    if not all_files:
        status_queue.put({"status": "ERROR", "msg": "No Data Found in ./data"})
        return

    pool_queue = queue.Queue(maxsize=3)
    loader_thread = threading.Thread(
        target=background_loader_manager, 
        args=(all_files, start_file_ptr, pool_queue, stop_event, cfg)
    )
    loader_thread.daemon = True
    loader_thread.start()
    
    # 4. RUNTIME VARIABLES
    current_epoch = start_epoch
    global_file_ptr = start_file_ptr
    pool_counter = 0
    
    logger.info(">>> ENGINE READY. WAITING FOR DATA STREAM...")

    # =========================================================================
    # --- INFINITE TRAINING LOOP (COMPLETE & FIXED) ---
    # =========================================================================
    while not stop_event.is_set() and current_epoch < cfg["EPOCHS"]:
        
        # --- A. SAFETY CHECKS ---
        if check_panic_switches(cfg["MIN_SYS_RAM_GB"]):
            time.sleep(5)
            continue
            
        # --- B. DYNAMIC CURRICULUM (THE BRAIN) ---
        target_perc = cfg["PERCEPTUAL_WEIGHT"]
        target_beta = cfg["BETA"]
        
        # Default Weights
        curr_mse = cfg["MSE_WEIGHT"]
        curr_edge = 0.0
        curr_perc = target_perc
        curr_beta = target_beta
        phase_name = "Standard"

        # PHASE DEFINITIONS
        if current_epoch < 5:
            phase_name = "PHASE 0: WARMUP"
            curr_perc = 0.0
            curr_beta = 0.0
            
        elif current_epoch < 30:
            phase_name = "PHASE 1: TEXTURE"
            curr_perc = target_perc
            curr_beta = 0.0001
            
        elif current_epoch < 150:
            phase_name = "PHASE 2: COMPRESSION"
            curr_beta = target_beta
            
        else:
            # === THE SHARPNESS PHASE (Refined for Epoch 250+) ===
            phase_name = "PHASE 3: AGGRESSIVE TEXT FOCUS"
            
            # 1. EDGE WEIGHT: The nuclear option. 
            # We push this high to force the gradients to snap to the text boundaries.
            curr_edge = 40.0 
            
            # 2. PIXEL WEIGHT: Absolute fealty to the input pixels.
            curr_mse = cfg["MSE_WEIGHT"] * 3.0
            
            # 3. REDUCE TEXTURE: VGG often fights sharpness by trying to preserve "grain".
            # We cut it by 75% to clean up the background.
            curr_perc = target_perc * 0.25
            
            # 4. BETA: Kill the VAE noise almost entirely.
            curr_beta = 1e-9
            
        # --- C. LR RESUSCITATION (THE FIX) ---
        # If we are deep in training and LR is dead, wake it up.
        current_lr = optimizer.param_groups[0]['lr']
        if current_epoch > 150 and current_lr < 1e-7:
            new_lr = 1e-4 # Reset to safe training speed
            logger.info(f" >>> PHASE CHANGE DETECTED: LR Resuscitated to {new_lr}")
            
            # 1. Force update optimizer
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
                
            # 2. Reset Scheduler so it doesn't kill it immediately
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                 optimizer, mode='min', factor=0.5, patience=10, verbose=True
            )

        # --- D. DATA FETCH ---
        try:
            # Wait for loader thread
            pool_data_cpu, new_ptr = pool_queue.get(timeout=2)
        except queue.Empty:
            continue
        
        # Move to GPU
        pool_data = pool_data_cpu.to(device, non_blocking=True)
        del pool_data_cpu
        
        dataset_size = pool_data.size(0)
        
        # --- E. TRAINING BATCH LOOP ---
        model.train()
        indices = torch.randperm(dataset_size)
        
        total_pool_loss = 0
        batches_processed = 0
        
        optimizer.zero_grad(set_to_none=True)
        
        for i in range(0, dataset_size, cfg["BATCH_SIZE"]):
            if stop_event.is_set(): break
            
            # Slice Batch
            batch = pool_data[indices[i : i+cfg["BATCH_SIZE"]]]
            
            # Skip incomplete batches
            if batch.size(0) < 2: continue
            
            # --- FORWARD PASS (AMP) ---
            with torch.amp.autocast('cuda'):
                recon, mu, logvar = model(batch)
                
                # UPDATED LOSS CALL
                # Passing 'curr_edge' which is now 15.0 in Phase 3
                loss = vae_loss_function(
                    recon, batch, mu, logvar,
                    beta=curr_beta,
                    mse_weight=curr_mse,
                    edge_weight=curr_edge,
                    chroma_weight=0.0,
                    perceptual_weight=curr_perc
                )
                
                loss = loss / cfg["GRAD_ACCUM_STEPS"]

            # --- BACKWARD PASS ---
            if torch.isnan(loss):
                logger.warning(" [!] NaN Loss Detected. Skipping Batch.")
                optimizer.zero_grad(set_to_none=True)
                continue
            
            # Calculate Gradients
            scaler.scale(loss).backward()
            
            # Gradient Accumulation Step
            if (i // cfg["BATCH_SIZE"]) % cfg["GRAD_ACCUM_STEPS"] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            
            # Stats
            loss_val = loss.item() * cfg["GRAD_ACCUM_STEPS"]
            total_pool_loss += loss_val
            batches_processed += 1
            
            # --- LIVE GUI UPDATES ---
            if batches_processed % 25 == 0:
                status_queue.put({
                    "status": "TRAINING",
                    "epoch": current_epoch,
                    "loss": loss_val,
                    "lr": optimizer.param_groups[0]['lr'],
                    "perc": curr_edge, # VISUAL HACK: Show Edge Weight in the GUI
                    "beta": curr_beta
                })

        # --- F. END OF POOL MANAGEMENT ---
        if batches_processed > 0:
            avg_pool_loss = total_pool_loss / batches_processed
            
            # Scheduler Step
            scheduler.step(avg_pool_loss)
            
            # 1. Checkpoint
            if pool_counter % cfg["CHECKPOINT_EVERY_N_POOLS"] == 0:
                save_titan_checkpoint(model, optimizer, current_epoch, new_ptr, avg_pool_loss, cfg)
            
            # 2. Visualize
            if pool_counter % cfg["VISUALIZE_EVERY_N_POOLS"] == 0:
                with torch.no_grad():
                    n_viz = min(4, batch.size(0))
                    comp = torch.cat([batch[:n_viz], recon[:n_viz]])
                    save_path = f"{cfg['RESULTS_DIR']}/e{current_epoch}_p{pool_counter}.png"
                    save_image(comp, save_path)
                    logger.info(f"SNAPSHOT SAVED: {save_path} [{phase_name}]")

        # --- G. CLEANUP & PROGRESS ---
        del pool_data, batch, recon, mu, logvar, loss
        cleanup_memory()
        
        # Check Epoch Rollover
        if new_ptr < global_file_ptr: 
            current_epoch += 1
            logger.info(f"=== EPOCH {current_epoch} COMPLETED ===")
            
        global_file_ptr = new_ptr
        pool_counter += 1
        
        # Heartbeat
        with open(os.path.join(cfg["CHECKPOINT_DIR"], "heartbeat.txt"), 'w') as f:
            f.write(f"Epoch: {current_epoch}, Time: {time.time()}")

    # Finish
    logger.info("TRAINING COMPLETE.")
    status_queue.put({"status": "STOPPED", "msg": "Max Epochs Reached"})

# =============================================================================
# --- SAFE LAUNCH WRAPPER ---
# =============================================================================
def run_process_safe_entry(status_queue, stop_event, config):
    """
    Called by titan_gui.py.
    Wraps the whole process in a global try/except block to ensure 
    errors are sent back to the GUI before crashing.
    """
    try:
        # Re-assert Env Vars for Safety
        os.environ["OMP_NUM_THREADS"] = "1"
        
        train_process_entrypoint(status_queue, stop_event, config)
        
    except Exception as e:
        # Save Crash Report
        with open("CRASH_REPORT.log", "w") as f:
            f.write(traceback.format_exc())
            
        err_msg = f"CRITICAL PROCESS FAILURE:\n{str(e)}\n{traceback.format_exc()}"
        print(err_msg)
        status_queue.put({"status": "ERROR", "msg": err_msg})
        
if __name__ == "__main__":
    print("This module is designed to be run via titan_gui.py")