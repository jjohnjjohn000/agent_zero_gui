"""
=============================================================================
TITAN ENGINE v4.0 - VISION TRAINING SYSTEM (ULTRA ROBUST & THREADED)
=============================================================================
Conçu pour l'entraînement haute performance sur GPU AMD (ROCm) avec 12 Go VRAM.
Optimisé pour le traitement de datasets massifs (21 Go+) par segmentation 
de mémoire et combustion directe en VRAM.

Dernière mise à jour : Intégration du Multi-Threading I/O et Vectorisation NumPy.
Adapte pour Multiprocessing (GUI Safe).
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

# --- VÉRIFICATION DES DÉPENDANCES CRITIQUES ---
# Ces imports sont faits ici pour que le script puisse être validé syntaxiquement
# même sans environnement complet (pour le GUI).
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

# =============================================================================
# --- CONFIGURATION DU MATÉRIEL (AMD / ROCm) ---
# =============================================================================
# Ces paramètres seront ré-appliqués à l'intérieur du processus d'entraînement
os.environ['MIOPEN_DISABLE_CACHE'] = '1'
os.environ['PYTORCH_HIP_ALLOC_CONF'] = 'max_split_size_mb:128'
os.environ['MIOPEN_FIND_MODE'] = '1'
os.environ['MIOPEN_DEBUG_DISABLE_CONV_ALGO_TYPES'] = '1'

# Injection du chemin projet pour les imports locaux
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_dir
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import du modèle VAE
try:
    from vae import VAE, vae_loss_function
except ImportError:
    try:
        from models.vae import VAE, vae_loss_function
    except:
        pass

# =============================================================================
# --- DEFAULTS & GLOBALS ---
# =============================================================================
DEFAULT_CONFIG = {
    "LEARNING_RATE": 5e-5,
    "BATCH_SIZE": 4,
    "LATENT_DIM": 4096,
    "CROP_SIZE": 256,
    "EPOCHS": 150,
    "NUM_THREADS": 6,
    "MAX_IMAGES_PER_POOL": 4000,
    "MIN_SYS_RAM_GB": 4.0,
    "FILES_PER_LOAD": 10,
    "PERCEPTUAL_WEIGHT": 1.0, # NEW: Replaces Edge Weight
    "EDGE_WEIGHT": 0.0,       # Deprecated
    "CHROMA_WEIGHT": 30.0,
    "BETA": 0.00001,
    "VISUALIZE_EVERY_N_POOLS": 2,
    "CHECKPOINT_EVERY_N_POOLS": 1,
    "DATA_DIR": os.path.join(project_root, "data"),
    "CHECKPOINT_DIR": os.path.join(project_root, "checkpoints"),
    "RESULTS_DIR": os.path.join(project_root, "results")
}

logger = logging.getLogger("TITAN")

# =============================================================================
# --- CLASSES ET UTILITAIRES DE SÉCURITÉ ---
# =============================================================================

class InMemoryDataset(Dataset):
    """Encapsulation des données résidentes en RAM pour PyTorch."""
    def __init__(self, tensor_data):
        self.data = tensor_data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

def get_vram_usage():
    """Mesure chirurgicale de l'occupation mémoire du GPU en Go."""
    if 'torch' in sys.modules and torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024**3)
    return 0

def cleanup_memory():
    """Libération forcée des ressources mémoire (Python & CUDA)."""
    gc.collect()
    if 'torch' in sys.modules and torch.cuda.is_available():
        torch.cuda.empty_cache()

def check_panic_switches(min_sys_ram):
    """Moniteur de sécurité système en temps réel."""
    mem = psutil.virtual_memory()
    avail_sys_gb = mem.available / (1024**3)
    
    if avail_sys_gb < min_sys_ram:
        logger.critical(f"PANIC RAM SYSTÈME : Seulement {avail_sys_gb:.2f} Go restants. Arrêt de sécurité.")
        return True
    
    if torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
        if peak_vram > 11.6:
            logger.critical(f"PANIC VRAM : Pic de consommation à {peak_vram:.2f} Go détecté.")
            return True
            
    return False

atexit.register(cleanup_memory)

# =============================================================================
# --- DATA LOADING ENGINE (THREADED & VECTORIZED) ---
# =============================================================================

def load_single_file(args):
    """
    SMART LOADER (Universal):
    - If image is huge (1080p): Randomly crop to 256x256 & Flip BGR->RGB.
    - If image is 256x256: Use full frame & Keep colors (assumes pre-processed).
    """
    fp, crop_size = args
    try:
        with np.load(fp) as data:
            if 'images' not in data:
                return None
            
            # Load data
            raw_images = data['images'] # Shape: (N, H, W, 3)
            n, h, w, c = raw_images.shape
            
            # --- CASE 1: PRE-PROCESSED (New Recorder) ---
            # The image is already the perfect size (256x256).
            # We assume new recorder saves RGB, so NO color flipping needed.
            if h == crop_size and w == crop_size:
                return raw_images.copy()

            # --- CASE 2: RAW HIGH-RES (Old Recorder) ---
            # The image is big. We need to crop a 256x256 patch.
            # We assume old recorder saved raw OpenCV (BGR), so we FLIP to RGB.
            elif h > crop_size and w > crop_size:
                # Vectorized Random Cropping
                y = np.random.randint(0, h - crop_size, size=n)
                x = np.random.randint(0, w - crop_size, size=n)
                
                patches = np.zeros((n, crop_size, crop_size, 3), dtype=np.uint8)
                for i in range(n):
                    patches[i] = raw_images[i, y[i]:y[i]+crop_size, x[i]:x[i]+crop_size]
                
                # Flip BGR -> RGB for the AI
                return patches[:, :, :, ::-1].copy()
            
            # --- CASE 3: TOO SMALL (Corrupt) ---
            else:
                return None

    except Exception as e:
        return None

def background_loader_manager(all_files, start_ptr, pool_queue, stop_event, cfg):
    """
    Gère le thread principal de chargement qui distribue le travail.
    """
    current_ptr = start_ptr
    total_files = len(all_files)
    
    with ThreadPoolExecutor(max_workers=cfg["NUM_THREADS"]) as executor:
        while not stop_event.is_set():
            vram_resident_pool = []
            current_pool_size = 0
            
            while current_pool_size < cfg["MAX_IMAGES_PER_POOL"] and not stop_event.is_set():
                batch_end_ptr = min(current_ptr + cfg["FILES_PER_LOAD"], total_files)
                files_to_open = all_files[current_ptr : batch_end_ptr]
                
                if not files_to_open:
                    current_ptr = 0  # Rebouclage infini sur le dataset
                    continue
                
                # Prep arguments
                args_list = [(f, cfg["CROP_SIZE"]) for f in files_to_open]
                results = list(executor.map(load_single_file, args_list))
                
                for res in results:
                    if res is not None:
                        # Conversion NumPy -> Tenseur CPU (Float32)
                        cpu_tensor = torch.from_numpy(
                            res.astype(np.float32) / 255.0
                        ).permute(0, 3, 1, 2)
                        
                        vram_resident_pool.append(cpu_tensor)
                        current_pool_size += res.shape[0]
                
                current_ptr = batch_end_ptr

            if vram_resident_pool and not stop_event.is_set():
                # Fusion finale avant envoi au GPU
                full_pool = torch.cat(vram_resident_pool, dim=0)
                try:
                    pool_queue.put((full_pool, current_ptr), timeout=5)
                except queue.Full:
                    pass

# =============================================================================
# --- PERSISTANCE DES ÉTATS (SAVE/LOAD) ---
# =============================================================================

def save_titan_checkpoint(model, optimizer, epoch, file_ptr, loss, cfg):
    """Sauvegarde les poids du modèle et l'état de progression JSON."""
    weights_path = os.path.join(cfg["CHECKPOINT_DIR"], "vae_latest.pth")
    state_path = os.path.join(cfg["CHECKPOINT_DIR"], "training_state.json")
    
    # 1. Sauvegarde des poids neuronaux
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, weights_path)

    # 2. Sauvegarde des métadonnées de session
    state = {
        "timestamp": datetime.now().isoformat(),
        "epoch": epoch, 
        "file_index": file_ptr, 
        "last_loss": float(loss),
        "tuning": {
            "lr": optimizer.param_groups[0]['lr']
        }
    }
    with open(state_path, 'w') as f:
        json.dump(state, f, indent=4)
    logger.info(f" >>> [SAVE] Checkpoint : Epoch {epoch}, File {file_ptr}, Loss {loss:.4f}")

def load_titan_checkpoint(model, optimizer, cfg):
    """Restaure la session d'entraînement précédente."""
    weights_path = os.path.join(cfg["CHECKPOINT_DIR"], "vae_latest.pth")
    state_path = os.path.join(cfg["CHECKPOINT_DIR"], "training_state.json")
    start_epoch, start_file_ptr = 0, 0
    
    if os.path.exists(weights_path):
        try:
            checkpoint = torch.load(weights_path, map_location='cpu')
            # Try loading weights. If architecture changed (V3->V4), this will fail.
            try:
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                logger.info(" >>> [LOAD] Poids neuronaux restaurés.")
            except RuntimeError as e:
                logger.warning(f" >>> [ARCH CHANGE DETECTED] Weights mismatch (V4 Upgrade). Starting Fresh. {e}")
                return 0, 0
        except Exception as e:
            logger.error(f"Échec restauration poids : {e}")

    if os.path.exists(state_path):
        try:
            with open(state_path, 'r') as f:
                state_data = json.load(f)
                start_epoch = state_data.get("epoch", 0)
                start_file_ptr = state_data.get("file_index", 0)
                logger.info(f" >>> [LOAD] Session reprise à : Epoch {start_epoch}, Fichier {start_file_ptr}")
        except:
            logger.warning("Fichier d'état illisible. Reprise à zéro.")
            
    return start_epoch, start_file_ptr

# =============================================================================
# --- MAIN PROCESS ENTRY POINT (GUI SAFE) ---
# =============================================================================

class QueueLogger(logging.Handler):
    """Redirige les logs vers la Queue du GUI"""
    def __init__(self, queue):
        super().__init__()
        self.queue = queue
        self.formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

    def emit(self, record):
        try:
            msg = self.format(record)
            self.queue.put({"status": "LOG", "msg": msg})
        except:
            self.handleError(record)

def train_process_entrypoint(status_queue, stop_event, config_overrides):
    """
    Fonction principale exécutée dans un PROCESSUS SÉPARÉ.
    C'est ici que tout se passe.
    """
    
    local_stop_event = threading.Event() 

    # 1. CONFIGURATION
    cfg = DEFAULT_CONFIG.copy()
    if config_overrides:
        cfg.update(config_overrides)
        
    # Création des infrastructures physiques
    os.makedirs(cfg["CHECKPOINT_DIR"], exist_ok=True)
    os.makedirs(cfg["RESULTS_DIR"], exist_ok=True)
    
    # 2. LOGGING SETUP
    LOG_FILE = os.path.join(cfg["CHECKPOINT_DIR"], f"titan_train_{datetime.now().strftime('%Y%m%d_%H%M')}.log")
    
    logger.setLevel(logging.INFO)
    logger.handlers = [] # Clean slate for new process
    
    # File Handler
    fh = logging.FileHandler(LOG_FILE)
    fh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(fh)
    
    # GUI Queue Handler
    qh = QueueLogger(status_queue)
    logger.addHandler(qh)
    
    logger.info("TITAN ENGINE V4.0 STARTED.")
    
    try:
        # 3. INITIALIZATION MATÉRIELLE (DELAYED IMPORT)
        # Import local pour éviter les crashs si importé globalement
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"TITAN ENGINE INITIALISÉ SUR : {device}")
        
        # 4. CONSTRUCTION DU MODÈLE
        model = VAE(latent_dim=cfg["LATENT_DIM"]).to(device)
        optimizer = optim.Adam(model.parameters(), lr=cfg["LEARNING_RATE"])
        
        # 5. RESTAURATION DE SESSION
        start_epoch, start_file_ptr = load_titan_checkpoint(model, optimizer, cfg)
        
        # --- CRITICAL FIX: FORCE GUI LEARNING RATE ---
        # Overwrite the checkpoint's stored LR with the fresh value from the GUI
        current_gui_lr = cfg["LEARNING_RATE"]
        logger.info(f" >>> [INIT] Overriding LR to GUI Value: {current_gui_lr}")
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_gui_lr
        
        # Sanity Check
        logger.info(">>> Lancement du Sanity Check...")
        try:
            dummy = torch.randn(1, 3, cfg["CROP_SIZE"], cfg["CROP_SIZE"]).to(device)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                recon, mu, logvar = model(dummy)
                # FIX: Added 9th argument (perceptual_weight=0.0)
                loss = vae_loss_function(recon, dummy, mu, logvar, 0.0, 0.0, 0.0, 0.0, 0.0)
            loss.backward()
            optimizer.zero_grad(set_to_none=True)
            logger.info(">>> Sanity Check RÉUSSI.")
        except Exception as e:
            logger.critical(f"ÉCHEC DU SANITY CHECK : {e}")
            status_queue.put({"status": "ERROR", "msg": str(e)})
            return # Fatal error

        # Scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        # FIX: Set a lower initial scale and lower growth factor
        scaler = torch.amp.GradScaler('cuda', init_scale=1024.0, growth_interval=2000)
        torch.backends.cudnn.benchmark = True 
        
        # 6. PRÉPARATION DU DATASET
        data_pattern = os.path.join(cfg["DATA_DIR"], "*.npz")
        all_files = sorted(glob.glob(data_pattern))
        if not all_files:
            logger.error(f"Données introuvables dans {cfg['DATA_DIR']}.")
            status_queue.put({"status": "ERROR", "msg": "No Data Found"})
            return

        # 7. VARIABLES DE SESSION
        pool_queue = queue.Queue(maxsize=2)
        local_stop_event = threading.Event()
        pool_counter = 0
        global_file_ptr = start_file_ptr
        current_epoch = start_epoch
        last_loop_ptr = start_file_ptr
        
        # Lancement du loader
        loader_thread = threading.Thread(
            target=background_loader_manager, 
            args=(all_files, global_file_ptr, pool_queue, local_stop_event, cfg)
        )
        loader_thread.daemon = True
        loader_thread.start()
        
        status_queue.put({"status": "RUNNING", "msg": "Training Loop Active"})
        
        # --- PHASE CONTROL FLAGS ---
        phases_reset_done = {5: False, 15: False, 30: False}

        # =============================================================================
        # 8. BOUCLE MAITRESSE (THE INFINITE BURN)
        # =============================================================================
        pool_counter = 0
        last_loop_ptr = start_file_ptr
        phases_reset_done = {5: False, 15: False, 30: False}

        while not stop_event.is_set() and current_epoch < cfg["EPOCHS"]:
            # --- SAFETY CHECK ---
            if check_panic_switches(cfg["MIN_SYS_RAM_GB"]): 
                logger.critical("SYSTEM SAFETY TRIGGERED: Low Resources. Emergency Shutdown.")
                break
            
            # =================================================================
            # SCHEDULER V6.0: THE "TYPESETTER" RAMP (Fixed Text Signal)
            # =================================================================
            # Rationale: 
            # 1. BANNED Perceptual Loss until Epoch 30. It causes text blurring.
            # 2. HELD MSE High (100.0) much longer. This forces high contrast.
            # 3. LINEAR RAMP for Edge Loss. No more "shock" transitions.
            
            target_perc = cfg["PERCEPTUAL_WEIGHT"] 
            target_beta = cfg["BETA"]

            if current_epoch < 5:
                # PHASE 0: BINARY ANCHOR (Pure Contrast)
                # Forces the model to learn Black vs White pixels immediately.
                curr_mse, curr_chroma, curr_edge, curr_perc, curr_beta = 300.0, 50.0, 0.0, 0.0, 0.0
                phase_name = "PHASE 0: ANCHOR"

            elif current_epoch < 30:
                # PHASE 1: THE TYPESETTER (Linear Edge Ramp)
                # We ramp Edge loss slowly from 0 -> 150. 
                # We KEEP MSE at 100.0 to prevent "graying out" of text.
                # STRICTLY NO PERCEPTUAL LOSS yet.
                progress = (current_epoch - 5) / 25.0  # 0.0 to 1.0
                
                curr_mse = 100.0
                curr_chroma = 50.0
                curr_edge = 10.0 + (140.0 * progress) # Ramp: 10 -> 150
                curr_perc = 0.0 
                curr_beta = 0.0
                phase_name = f"PHASE 1: TYPESET (Edge: {curr_edge:.1f})"

            elif current_epoch < 50:
                # PHASE 2: LATENT ORGANIZATION (Beta Warmup)
                # Text is now sharp. We slowly introduce Beta to organize the latent space.
                # We slowly lower MSE to 20 to allow for more artistic freedom later.
                progress = (current_epoch - 30) / 20.0
                
                curr_mse = 100.0 - (80.0 * progress) # Drop: 100 -> 20
                curr_chroma = 50.0
                curr_edge = 150.0 # Hold Edge high
                curr_perc = 0.0   # Still no VGG
                curr_beta = target_beta * progress # Ramp: 0 -> Target
                phase_name = "PHASE 2: COMPRESS"

            else:
                # PHASE 3: TEXTURE FILL (VGG Integration)
                # Text structure is locked in. Now we add VGG to fix blurry backgrounds.
                curr_mse = 20.0
                curr_chroma = 50.0
                curr_edge = 150.0
                curr_perc = target_perc
                curr_beta = target_beta
                phase_name = "PHASE 3: FINAL"

            # --- AMP SAFETY ---
            # Disable Mixed Precision during Phase 0/1 to prevent NaN on the high MSE
            use_amp = (current_epoch >= 40)

            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.0001 if current_epoch < 30 else cfg["LEARNING_RATE"]
            
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f"\n--- EPOCH {current_epoch} | {phase_name} | AMP: {use_amp} ---")

            # =================================================================
            # PHASE A : RAVITAILLEMENT (GET DATA FROM RAM -> VRAM)
            # =================================================================
            try:
                # Wait for the background loader thread to provide a data pool
                res = None
                while not stop_event.is_set():
                    try:
                        res = pool_queue.get(timeout=2)
                        break
                    except queue.Empty:
                        continue
                
                if res is None or stop_event.is_set(): break
                
                current_vram_data_cpu, global_file_ptr = res
                # Crucial: Move to GPU in one block
                current_vram_data = current_vram_data_cpu.to(device, non_blocking=True)
                del current_vram_data_cpu # Clear CPU memory immediately
            except Exception as e:
                logger.warning(f"Data Pipeline Error: {e}")
                continue

            # =================================================================
            # PHASE B : LA COMBUSTION (THE ACTUAL TRAINING LOOP)
            # =================================================================
            num_pool_samples = len(current_vram_data)
            cycle_loss_accumulator = 0
            cycle_iterations = 0
            
            indices = torch.randperm(num_pool_samples)
            model.train()
            
            for i in range(0, num_pool_samples, cfg["BATCH_SIZE"]):
                if stop_event.is_set(): break
                
                batch_indices = indices[i : i + cfg["BATCH_SIZE"]]
                batch = current_vram_data[batch_indices]
                
                if batch.size(0) < 2: continue

                # --- [START ROBUST TRAINING STEP] ---
                optimizer.zero_grad(set_to_none=True)
                
                # On force le Float32 tant qu'on est pas en Phase 3
                use_amp = (current_epoch >= 41) 
                
                # 1. Forward Pass
                if use_amp:
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        recon, mu, logvar = model(batch)
                        loss = vae_loss_function(recon, batch, mu, logvar, curr_beta, curr_mse, curr_edge, curr_chroma, curr_perc)
                else:
                    # Float32 pur pour la stabilité avec les gros poids
                    recon, mu, logvar = model(batch)
                    loss = vae_loss_function(recon, batch, mu, logvar, curr_beta, curr_mse, curr_edge, curr_chroma, curr_perc)

                # --- FIX START: Detect Broken Graph ---
                if loss.item() == 1000.0:
                    logger.warning(" [!] NaN Detected (Graph Disconnected). Skipping Batch.")
                    optimizer.zero_grad(set_to_none=True)
                    continue

                # 2. Backward Pass & Step
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # En Float32, on n'utilise JAMAIS le scaler
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                
                cycle_loss_accumulator += loss.item()
                cycle_iterations += 1
                
                # Update GUI with live loss every few steps
                if cycle_iterations % 20 == 0:
                    status_queue.put({
                        "status": "TRAINING",
                        "epoch": current_epoch,
                        "loss": loss.item(),
                        "lr": current_lr,
                        "perc": curr_perc
                    })

            # =================================================================
            # PHASE C : SYNCHRONISATION & PERSISTANCE
            # =================================================================
            if not stop_event.is_set() and cycle_iterations > 0:
                avg_pool_loss = cycle_loss_accumulator / cycle_iterations
                scheduler.step(avg_pool_loss)
                
                # Checkpointing
                if pool_counter % cfg["CHECKPOINT_EVERY_N_POOLS"] == 0:
                    save_titan_checkpoint(model, optimizer, current_epoch, global_file_ptr, avg_pool_loss, cfg)
                
                # Visualization
                if pool_counter % cfg["VISUALIZE_EVERY_N_POOLS"] == 0:
                    with torch.no_grad():
                        n_viz = min(batch.size(0), 8)
                        visual_comparison = torch.cat([batch[:n_viz], recon[:n_viz]])
                        snap_path = os.path.join(cfg["RESULTS_DIR"], f"v_e{current_epoch}_p{pool_counter}.png")
                        save_image(visual_comparison.cpu(), snap_path, nrow=n_viz)
                        logger.info(f" [!] Snapshot saved: {snap_path}")

            # =================================================================
            # PHASE D : AGGRESSIVE PURGE & EPOCH MANAGEMENT
            # =================================================================
            # 1. Clear GPU variables immediately
            try:
                # We target every large tensor that could be lingering
                del batch, recon, mu, logvar, current_vram_data, loss
            except NameError:
                pass 

            # 2. Hard memory reset for ROCm/CUDA stability
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect() # Clears shared memory fragments

            # 3. Epoch increment logic (Check if dataset pointer wrapped)
            if global_file_ptr < last_loop_ptr:
                current_epoch += 1
                logger.info(f" >>> [EPOCH ROLLOVER] New Epoch: {current_epoch}")
            
            last_loop_ptr = global_file_ptr
            pool_counter += 1
            
            # Heartbeat (Indicator that the engine is still alive)
            with open(os.path.join(cfg["CHECKPOINT_DIR"], "heartbeat.txt"), "w") as f:
                f.write(f"Epoch: {current_epoch}, Pool: {pool_counter}, Time: {time.time()}")
            
    except Exception as e:
        # Emergency Disk Log
        with open("CRITICAL_CRASH.log", "a") as f:
            f.write(f"\n[{datetime.now()}] CRASH: {str(e)}\n")
            f.write(traceback.format_exc())
        
        logger.error(f"CRITICAL ENGINE FAILURE: {e}")
        status_queue.put({"status": "ERROR", "msg": str(e)})
        
    finally:
        logger.info(" >>> ARRÊT DU MOTEUR TITAN : Nettoyage Final...")
        local_stop_event.set()
        status_queue.put({"status": "STOPPED", "msg": "Process Ended"})

# =============================================================================
# MULTIPROCESSING SAFE ENTRY POINT (ADD THIS AT THE END)
# =============================================================================
def run_process_safe_entry(status_queue, stop_event, config):
    """
    Called by the GUI process. Runs strictly inside the child process.
    Ensures environment variables are set before libraries load.
    """
    import os
    import sys
    import traceback

    try:
        # Re-apply critical env vars for the subprocess
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1" 
        os.environ['MIOPEN_DISABLE_CACHE'] = '1'
        os.environ['PYTORCH_HIP_ALLOC_CONF'] = 'max_split_size_mb:128'
        
        # Start the training logic
        train_process_entrypoint(status_queue, stop_event, config)

    except Exception as e:
        # Catch imports or startup errors
        try:
            status_queue.put({
                "status": "ERROR", 
                "msg": f"Process Launch Error: {e}\n{traceback.format_exc()}"
            })
        except:
            print(f"CRITICAL SUBPROCESS FAILURE: {e}")

if __name__ == "__main__":
    print("This module is intended to be run via titan_gui.py")