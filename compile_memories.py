import os
import glob
import numpy as np
import torch
import sys

# --- CONFIGURATION ---
RAW_DATA_DIR = "./data"
MEMORY_DIR = "./data/memories"
VAE_CHECKPOINT = "./checkpoints/vae_latest.pth"
BATCH_SIZE = 64  # Fast processing

# Ensure models can be imported
sys.path.append(os.getcwd())

try:
    from models.vae import VAE
except ImportError:
    try:
        from vae import VAE
    except ImportError:
        print("CRITICAL: Could not import 'vae.py'.")
        sys.exit(1)

def compile():
    # 1. Setup
    if not os.path.exists(RAW_DATA_DIR):
        print(f"ERROR: '{RAW_DATA_DIR}' not found.")
        return

    raw_files = sorted(glob.glob(os.path.join(RAW_DATA_DIR, "*.npz")))
    if not raw_files:
        print(f"ERROR: No recordings found in '{RAW_DATA_DIR}'.")
        return

    os.makedirs(MEMORY_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"-> Found {len(raw_files)} recordings.")
    
    # 2. Load Vision Model (The Eyes)
    print("-> Loading VAE...")
    # Note: We enforce 256 size here to match your recorder
    vae = VAE(latent_dim=4096, img_size=256).to(device)
    
    try:
        checkpoint = torch.load(VAE_CHECKPOINT, map_location=device)
        vae.load_state_dict(checkpoint['model_state_dict'])
    except Exception as e:
        print(f"Error loading VAE weights: {e}")
        return

    vae.eval()

    # 3. Digestion Loop
    print("-> Digesting visual data into memories...")
    count = 0
    
    for f_idx, fpath in enumerate(raw_files):
        fname = os.path.basename(fpath)
        save_path = os.path.join(MEMORY_DIR, f"mem_{fname}")
        
        if os.path.exists(save_path):
            continue

        try:
            with np.load(fpath) as data:
                if 'images' not in data: continue
                
                # Load Images (Already 256x256 from new recorder)
                raw_imgs = data['images'] 
                
                # Load Actions (Already [x, y, 0, 0, 0, 0])
                if 'actions' in data:
                    actions = data['actions']
                else:
                    # Fallback for very old files (shouldn't happen if you cleaned data)
                    actions = np.zeros((len(raw_imgs), 6))

                # ENCODE LOOP (Pixels -> Latents)
                latents_buffer = []
                with torch.no_grad():
                    for i in range(0, len(raw_imgs), BATCH_SIZE):
                        # Batching
                        batch = raw_imgs[i : i+BATCH_SIZE]
                        
                        # Normalize 0-255 -> 0.0-1.0
                        tensor = torch.from_numpy(batch).float().permute(0, 3, 1, 2) / 255.0
                        tensor = tensor.to(device)
                        
                        # Encode
                        mu, _ = vae.encode(tensor)
                        latents_buffer.append(mu.cpu().numpy())

                full_latents = np.concatenate(latents_buffer, axis=0)
                
                # Save Memory
                np.savez_compressed(save_path, latents=full_latents, actions=actions)
                count += 1
                print(f"[{f_idx+1}/{len(raw_files)}] Digested {fname} -> {full_latents.shape}")

        except Exception as e:
            print(f"Skipping {fname}: {e}")

    print(f"SUCCESS: Created {count} new memories.")

if __name__ == "__main__":
    compile()