"""
=============================================================================
TITAN ENGINE v3.1 - VISION TRAINING SYSTEM (ULTRA ROBUST & THREADED)
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
project_root = os.path.dirname(current_dir)
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
    "LEARNING_RATE": 8e-5,
    "BATCH_SIZE": 32,
    "LATENT_DIM": 2048,
    "CROP_SIZE": 128,
    "EPOCHS": 130,
    "NUM_THREADS": 6,
    "MAX_IMAGES_PER_POOL": 12000,
    "MIN_SYS_RAM_GB": 4.0,
    "FILES_PER_LOAD": 16,
    "EDGE_WEIGHT": 500.0,
    "CHROMA_WEIGHT": 30.0,
    "BETA": 0.0,
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
    Travailleur CPU Unitaire. 
    Note: args est un tuple (filepath, crop_size) pour compatibilité map.
    """
    fp, crop_size = args
    try:
        with np.load(fp) as data:
            if 'images' not in data:
                return None
            raw_images = data['images']
            
            n, h, w, c = raw_images.shape
            if h < crop_size or w < crop_size:
                return None
            
            # Stratégie de data-augmentation (Crops multiples vectorisés)
            y = np.random.randint(0, h - crop_size, size=n)
            x = np.random.randint(0, w - crop_size, size=n)
            
            # Allocation d'un bloc contigu pour la rapidité
            patches = np.zeros((n, crop_size, crop_size, 3), dtype=np.uint8)
            for i in range(n):
                patches[i] = raw_images[i, y[i]:y[i]+crop_size, x[i]:x[i]+crop_size]
            
            # CORRECTION : Conversion BGR -> RGB et retour
            return patches[:, :, :, ::-1].copy()
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
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info(" >>> [LOAD] Poids neuronaux restaurés.")
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
    
    logger.info("TITAN ENGINE PROCESS STARTED.")
    
    try:
        # 3. INITIALIZATION MATÉRIELLE (DELAYED IMPORT)
        # Import local pour éviter les crashs si importé globalement
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"TITAN ENGINE v3.1 INITIALISÉ SUR : {device}")
        
        # 4. CONSTRUCTION DU MODÈLE
        model = VAE(latent_dim=cfg["LATENT_DIM"]).to(device)
        optimizer = optim.Adam(model.parameters(), lr=cfg["LEARNING_RATE"])
        
        # 5. RESTAURATION DE SESSION
        start_epoch, start_file_ptr = load_titan_checkpoint(model, optimizer, cfg)
        
        # Sanity Check
        logger.info(">>> Lancement du Sanity Check...")
        try:
            dummy = torch.randn(1, 3, cfg["CROP_SIZE"], cfg["CROP_SIZE"]).to(device)
            with torch.amp.autocast('cuda'):
                recon, mu, logvar = model(dummy)
                loss = vae_loss_function(recon, dummy, mu, logvar, 0.0, 10.0, 10.0)
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
        scaler = torch.amp.GradScaler('cuda')
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
        
        phases_reset_done = {5: False, 15: False, 25: False} # Flags to ensure we reset only once

        # 8. BOUCLE MAITRESSE (THE INFINITE BURN)
        while not stop_event.is_set() and current_epoch < cfg["EPOCHS"]:
            if check_panic_switches(cfg["MIN_SYS_RAM_GB"]): break
            
            # --- LE SCHEDULER DE CONTRÔLE  ---
            # We use the GUI values as the "Target" (Max) values
            target_edge = cfg["EDGE_WEIGHT"]
            target_beta = cfg["BETA"]

            if current_epoch < 5:
                # Phase 1: 0% Edge, 1% Beta (Warmup)
                current_edge_weight = 0.0 
                current_beta = target_beta * 0.01
                phase_name = "PHASE 1 (COULEURS)"
            elif current_epoch < 15:
                # Phase 2: 50% Edge, 10% Beta
                current_edge_weight = target_edge * 0.5 
                current_beta = target_beta * 0.1
                phase_name = "PHASE 1.5 (BOOST STRUCTURE)"
            elif current_epoch < 25:
                # Phase 3: 100% Edge, 50% Beta
                current_edge_weight = target_edge
                current_beta = target_beta * 0.5
                phase_name = "PHASE 2 (STRUCTURE)"
            elif current_epoch < 130:
                # Phase 4: 100% Edge, 50% Beta
                current_edge_weight = 3000.0
                current_beta = 0.0
                phase_name = "PHASE 4 (FINISH)"
            else:
                # Phase 4: 100% Edge, 100% Beta (Full GUI Settings)
                current_edge_weight = target_edge
                current_beta = target_beta
                
                # Cap Learning Rate for fine-tuning phase
                for param_group in optimizer.param_groups:
                    if param_group['lr'] > 5e-5:
                        param_group['lr'] = 5e-5
                phase_name = "PHASE 3 (DETAILS)"
            
            # --- CRITICAL FIX: RESET SCHEDULER ON PHASE CHANGE ---
            # We must re-initialize the scheduler so it forgets the "low loss" from the previous phase.
            # Otherwise, it compares the new high loss (due to edge weights) to the old low loss and panics.
            if current_epoch in phases_reset_done and not phases_reset_done[current_epoch]:
                logger.info(f" >>> [PHASE CHANGE] RESETTING LR & SCHEDULER for Epoch {current_epoch}!")
                
                # 1. Reset Learning Rate to a strong value
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 8e-5
                
                # 2. Re-create Scheduler (clears 'best' loss history)
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='min', factor=0.5, patience=5, verbose=True
                )
                
                # 3. Mark as done so we don't do it for every pool in this epoch
                phases_reset_done[current_epoch] = True
            
            # --- LOG HYPERPARAMETERS ---
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f"\n--- EPOCH {current_epoch} | {phase_name} ---")
            logger.info(f" >>> [PARAMS] LR: {current_lr:.6f} | Beta: {current_beta:.6f} | Edge: {current_edge_weight:.1f} | Chroma: {cfg['CHROMA_WEIGHT']}")
            
            # --- PHASE A : RAVITAILLEMENT (RAM -> VRAM) ---
            try:
                # Wait for data with timeout to check stop_event
                while not stop_event.is_set():
                    try:
                        res = pool_queue.get(timeout=1)
                        break
                    except queue.Empty:
                        continue
                if stop_event.is_set(): break
                
                current_vram_data_cpu, global_file_ptr = res
                current_vram_data = current_vram_data_cpu.to(device, non_blocking=True)
                del current_vram_data_cpu
            except Exception as e:
                logger.warning(f"Pipeline wait error: {e}")
                continue

            # --- PHASE B : LA COMBUSTION (BURNING) ---
            num_pool_samples = len(current_vram_data)
            cycle_loss_accumulator = 0
            cycle_iterations = 0
            logger.info(f" >>> VRAM CHARGÉE : {num_pool_samples} images. EdgeWeight: {current_edge_weight}")
            
            indices = torch.randperm(num_pool_samples)
            model.train()
            
            for i in range(0, num_pool_samples, cfg["BATCH_SIZE"]):
                if stop_event.is_set() or check_panic_switches(cfg["MIN_SYS_RAM_GB"]): break
                
                batch_indices = indices[i : i + cfg["BATCH_SIZE"]]
                batch = current_vram_data[batch_indices]
                
                if batch.size(0) < 2: continue

                optimizer.zero_grad(set_to_none=True)
                
                with torch.amp.autocast('cuda'):
                    recon, mu, logvar = model(batch)
                    loss = vae_loss_function(
                        recon, batch, mu, logvar, 
                        beta=current_beta, 
                        edge_weight=current_edge_weight, 
                        chroma_weight=cfg["CHROMA_WEIGHT"]
                    )

                if torch.isnan(loss):
                    logger.error("\n [!] ALERTE : Perte NaN détectée.")
                    optimizer.zero_grad(set_to_none=True)
                    continue 

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
                
                cycle_loss_accumulator += loss.item()
                cycle_iterations += 1
                
                # Send Stats to GUI (Sampled)
                if cycle_iterations % 10 == 0:
                    status_queue.put({
                        "status": "TRAINING",
                        "epoch": current_epoch,
                        "iteration": cycle_iterations,
                        "loss": loss.item(),
                        "lr": optimizer.param_groups[0]['lr'],
                        "edge_weight": current_edge_weight,  # <--- Added to payload
                        "beta": current_beta,                # <--- Added to payload
                        "phase": phase_name
                    })

            # --- PHASE C : SYNCHRONISATION ET PERSISTANCE ---
            if not stop_event.is_set() and cycle_iterations > 0:
                avg_pool_loss = cycle_loss_accumulator / cycle_iterations
                
                # Update the GUI with the calculated average loss for this pool
                status_queue.put({"status": "TRAINING", "loss": avg_pool_loss, "epoch": current_epoch, "iteration": cycle_iterations, "lr": optimizer.param_groups[0]['lr']})

                logger.info(f"\n >>> Cycle Pool OK. Perte Moyenne : {avg_pool_loss:.4f}")
                
                # --- LOGIC FIX: MANUAL LR RESET ON PHASE CHANGE ---
                # If we just entered a new phase (based on epoch), we should BOOST the LR, not lower it.
                if current_epoch in [5, 15, 25] and pool_counter == 0:
                    logger.info(" >>> PHASE CHANGE DETECTED: Resetting Learning Rate to 8e-5")
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = 8e-5
                    # Reset scheduler to forget the "bad" history
                    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, mode='min', factor=0.5, patience=5, verbose=True
                    )
                else:
                    # Only step scheduler normally
                    scheduler.step(avg_pool_loss)
                
                if pool_counter % cfg["CHECKPOINT_EVERY_N_POOLS"] == 0:
                    save_titan_checkpoint(model, optimizer, current_epoch, global_file_ptr, avg_pool_loss, cfg)
                
                if pool_counter % cfg["VISUALIZE_EVERY_N_POOLS"] == 0:
                    with torch.no_grad():
                        n_viz = min(batch.size(0), 8)
                        visual_comparison = torch.cat([batch[:n_viz], recon[:n_viz]])
                        snap_path = os.path.join(cfg["RESULTS_DIR"], f"v_e{current_epoch}_p{pool_counter}.png")
                        save_image(visual_comparison.cpu(), snap_path, nrow=n_viz)
                        logger.info(f" [!] Snapshot Visuel généré : {snap_path}")

            # --- PHASE D : VIDANGE ET GESTION EPOCH ---
            del current_vram_data, batch, recon, mu, logvar
            cleanup_memory()
            
            if global_file_ptr < last_loop_ptr:
                current_epoch += 1
                logger.info(f" >>> [NEW EPOCH] Passage à l'EPOCH {current_epoch}")
            
            last_loop_ptr = global_file_ptr
            pool_counter += 1
            
    except Exception as e:
        logger.error(f"CRITICAL ENGINE FAILURE: {e}")
        logger.error(traceback.format_exc())
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