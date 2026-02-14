import os
import glob
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image

# Imports
try:
    from models.cortex import TitanCortex
    from models.vae import VAE
except ImportError:
    from cortex import TitanCortex
    from vae import VAE

# --- HYPERPARAMETERS ---
CONFIG = {
    "SEQ_LEN": 64,       
    "BATCH_SIZE": 16,     
    "EPOCHS": 100,
    "LR": 1e-4,          # Lowered slightly for stability
    "LATENT_DIM": 4096,
    "DATA_DIR": "./data/memories",
    "CHECKPOINT_DIR": "./checkpoints",
    "RESULTS_DIR": "./results_cortex"
}

os.makedirs(CONFIG["RESULTS_DIR"], exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- DATASET WITH NORMALIZATION ---
class MemoryDataset(Dataset):
    def __init__(self, seq_len):
        self.seq_len = seq_len
        self.files = sorted(glob.glob(os.path.join(CONFIG["DATA_DIR"], "*.npz")))
        self.data = []
        
        print(">>> Loading Memories...")
        all_latents = []
        
        # Load data
        for i, f in enumerate(self.files):
            try:
                with np.load(f) as d:
                    l = d['latents']
                    a = d['actions']
                    if len(l) > seq_len + 1:
                        self.data.append((l, a))
                        # Collect stats from first 50 files only (to save RAM)
                        if i < 50: all_latents.append(l)
            except: pass
            
        print(f">>> Loaded {len(self.data)} sequences.")

        # CALCULATE NORMALIZATION STATS
        if all_latents:
            concat = np.concatenate(all_latents, axis=0)
            self.mean = torch.from_numpy(concat.mean(axis=0)).float().to(device)
            self.std = torch.from_numpy(concat.std(axis=0)).float().to(device)
            # Avoid division by zero
            self.std[self.std < 1e-5] = 1.0
            print(f">>> Data Stats - Mean: {self.mean.mean():.4f} | Std: {self.std.mean():.4f}")
            print(">>> Normalization Enabled.")
        else:
            self.mean = torch.zeros(CONFIG["LATENT_DIM"]).to(device)
            self.std = torch.ones(CONFIG["LATENT_DIM"]).to(device)

    def __len__(self):
        return len(self.data) * 20

    def __getitem__(self, idx):
        file_idx = np.random.randint(len(self.data))
        latents, actions = self.data[file_idx]
        
        max_start = len(latents) - self.seq_len - 1
        start = np.random.randint(0, max_start)
        
        # Raw Data
        s_raw = torch.from_numpy(latents[start : start+self.seq_len+1]).float()
        a_in = torch.from_numpy(actions[start : start+self.seq_len]).float()
        
        return s_raw, a_in

# --- DREAM VISUALIZER (Corrected for Normalization) ---
# --- ENHANCED DREAM VISUALIZER (With Mouse Overlay) ---
def save_dream(model, vae, dataset, epoch):
    model.eval()
    vae.eval()
    
    # 1. Get Real Sequence
    # s_raw: (Seq+1, 4096), a_in: (Seq, 6)
    s_raw, a_in = dataset[0]
    
    # 2. Prepare Inputs
    s_in_raw = s_raw[:-1].to(device)
    s_in_norm = (s_in_raw - dataset.mean) / dataset.std
    
    a_in = a_in.to(device)
    
    # 3. Dream Loop (Predict Future States)
    s_in_norm = s_in_norm.unsqueeze(0)
    a_in_batch = a_in.unsqueeze(0)
    
    context_len = 10
    dream_state_norm = s_in_norm.clone()
    
    with torch.no_grad():
        for i in range(context_len, CONFIG["SEQ_LEN"] - 1):
            # The Brain predicts the next frame based on history
            preds = model(dream_state_norm[:, :i+1], a_in_batch[:, :i+1])
            next_val = preds[:, -1, :].unsqueeze(1)
            dream_state_norm[:, i+1, :] = next_val

    # 4. Decode to Images
    real_latents = s_in_raw
    dream_latents = (dream_state_norm[0] * dataset.std) + dataset.mean
    
    indices = range(0, CONFIG["SEQ_LEN"], 8)
    real_l = real_latents[indices]
    dream_l = dream_latents[indices]
    
    # Get corresponding mouse actions for these frames
    # Assuming Action format: [x, y, click, ...]
    # We assume x, y are normalized 0.0 to 1.0. If not, we might need to adjust.
    mouse_actions = a_in[indices, :2].cpu().numpy()
    
    with torch.no_grad():
        real_imgs = vae.decode(real_l).cpu()
        dream_imgs = vae.decode(dream_l).cpu()

    # 5. DRAW CURSORS (The New Part)
    def draw_cursor(imgs, coords):
        # imgs: (B, 3, 256, 256)
        # coords: (B, 2)
        h, w = imgs.shape[2], imgs.shape[3]
        for i, (mx, my) in enumerate(coords):
            # Convert normalized (0-1) to pixel (0-256)
            # If your recorder saved raw pixels (0-1920), we need to divide by screen size here.
            # Assuming normalized for now:
            px, py = int(mx * w), int(my * h)
            
            # Clamp to screen bounds
            px = max(2, min(w-3, px))
            py = max(2, min(h-3, py))
            
            # Draw Red Box (3x3)
            # Channel 0 = Red (1.0), Channel 1,2 = Green/Blue (0.0)
            imgs[i, 0, py-2:py+2, px-2:px+2] = 1.0 
            imgs[i, 1, py-2:py+2, px-2:px+2] = 0.0
            imgs[i, 2, py-2:py+2, px-2:px+2] = 0.0
        return imgs

    real_imgs = draw_cursor(real_imgs, mouse_actions)
    dream_imgs = draw_cursor(dream_imgs, mouse_actions)

    # 6. Save
    comparison = torch.cat([real_imgs, dream_imgs], dim=0)
    save_path = os.path.join(CONFIG["RESULTS_DIR"], f"dream_e{epoch}.png")
    save_image(comparison, save_path, nrow=len(indices))
    print(f"   >>> Dream saved with cursors: {save_path}")
    model.train()

# --- MAIN LOOP ---
def main():
    dataset = MemoryDataset(CONFIG["SEQ_LEN"])
    loader = DataLoader(dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=True, num_workers=0) # 0 workers for safety
    
    model = TitanCortex(state_dim=CONFIG["LATENT_DIM"]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["LR"], weight_decay=1e-5)
    
    # Load VAE for viz
    vae = VAE(latent_dim=CONFIG["LATENT_DIM"]).to(device)
    try:
        vae.load_state_dict(torch.load(os.path.join(CONFIG["CHECKPOINT_DIR"], "vae_latest.pth"))['model_state_dict'])
        print(">>> VAE Loaded.")
    except:
        vae = None

    print(">>> STARTING ROBUST TRAINING...")

    for epoch in range(CONFIG["EPOCHS"]):
        total_loss = 0
        
        for batch_idx, (s_raw, a_raw) in enumerate(loader):
            s_raw, a_raw = s_raw.to(device), a_raw.to(device)
            
            # 1. Normalize Inputs on the Fly
            # (Input - Mean) / Std
            s_norm = (s_raw - dataset.mean) / dataset.std
            
            # Split into Input and Target
            # Input: 0..63
            # Target: 1..64
            x = s_norm[:, :-1, :]
            y = s_norm[:, 1:, :]
            a = a_raw
            
            optimizer.zero_grad()
            
            # 2. Forward (Float32 for stability)
            pred = model(x, a)
            loss = nn.MSELoss()(pred, y)
            
            if torch.isnan(loss):
                print(f"!!! NAN DETECTED at Batch {batch_idx}. Skipping...")
                optimizer.zero_grad()
                continue
            
            loss.backward()
            
            # 3. Gradient Clipping (The Safety Net)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            total_loss += loss.item()
            
            if batch_idx % 50 == 0:
                print(f"   [Ep {epoch} | B {batch_idx}] Loss: {loss.item():.5f}")

        avg_loss = total_loss / len(loader)
        print(f"=== Epoch {epoch} Done | Avg Loss: {avg_loss:.5f} ===")
        
        # Save & Dream
        torch.save({'model_state_dict': model.state_dict()}, os.path.join(CONFIG["CHECKPOINT_DIR"], "cortex_latest.pth"))
        if epoch % 5 == 0 and vae:
            save_dream(model, vae, dataset, epoch)

if __name__ == "__main__":
    main()