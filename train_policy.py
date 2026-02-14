import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
import time
import sys

# --- IMPORTS ---
try:
    from models.policy import TitanPolicy
    from models.vae import VAE
except ImportError:
    sys.path.append(os.getcwd())
    from models.policy import TitanPolicy
    from models.vae import VAE

# --- CONFIG ---
CONFIG = {
    "BATCH_SIZE": 64,
    "EPOCHS": 50,
    "LR": 1e-4,
    "LATENT_DIM": 4096,
    "DATA_DIR": "./data/memories",
    "CHECKPOINT_DIR": "./checkpoints",
    "RESULTS_DIR": "./results_policy", # New folder for images
    "SAVE_NAME": "policy_latest.pth"
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(CONFIG["CHECKPOINT_DIR"], exist_ok=True)
os.makedirs(CONFIG["RESULTS_DIR"], exist_ok=True)

# --- ROBUST DATASET ---
class PolicyDataset(Dataset):
    def __init__(self):
        self.files = sorted(glob.glob(os.path.join(CONFIG["DATA_DIR"], "*.npz")))
        self.data_cache = []
        
        print(">>> [Dataset] Loading Behavioral Memories...")
        valid_frames = 0
        
        for f in self.files:
            try:
                with np.load(f) as d:
                    l = d['latents']
                    a = d['actions']
                    if len(l) == len(a) and len(l) > 10:
                        self.data_cache.append((l, a))
                        valid_frames += len(l)
            except: pass
        print(f">>> [Dataset] Loaded {len(self.data_cache)} sessions ({valid_frames} frames).")

    def __len__(self):
        if len(self.data_cache) == 0: return 0
        return len(self.data_cache) * 50

    def __getitem__(self, idx):
        session_idx = np.random.randint(len(self.data_cache))
        latents, actions = self.data_cache[session_idx]
        frame_idx = np.random.randint(0, len(latents))
        
        state = torch.from_numpy(latents[frame_idx]).float()
        target = torch.from_numpy(actions[frame_idx]).float()
        return state, target

# --- VISUALIZER ---
def save_target_practice(policy, vae, dataset, epoch):
    """
    Draws the "Mind's Eye" view of the screen.
    GREEN DOT = Where YOU moved the mouse (Ground Truth).
    RED DOT   = Where the AI wants to move (Prediction).
    """
    policy.eval()
    vae.eval()
    
    # Get 8 random samples
    states = []
    targets = []
    for _ in range(8):
        s, t = dataset[np.random.randint(len(dataset))]
        states.append(s)
        targets.append(t)
    
    states = torch.stack(states).to(device)
    targets = torch.stack(targets).to(device)
    
    with torch.no_grad():
        # 1. Get Prediction
        preds = policy(states)
        pred_xy = preds[:, :2]   # AI Mouse
        true_xy = targets[:, :2] # Human Mouse
        
        # 2. Decode Screen (So we can see what the AI sees)
        images = vae.decode(states)
        
        # 3. Draw Dots
        # Images are (B, 3, 256, 256)
        B, C, H, W = images.shape
        
        for i in range(B):
            # GREEN DOT (Human)
            gx = int(true_xy[i, 0].item() * W)
            gy = int(true_xy[i, 1].item() * H)
            gx, gy = max(0, min(W-1, gx)), max(0, min(H-1, gy))
            
            # RED DOT (AI)
            rx = int(pred_xy[i, 0].item() * W)
            ry = int(pred_xy[i, 1].item() * H)
            rx, ry = max(0, min(W-1, rx)), max(0, min(H-1, ry))
            
            # Draw 5x5 boxes
            # Green (Channel 1)
            images[i, 1, max(0,gy-2):gy+2, max(0,gx-2):gx+2] = 1.0 
            # Red (Channel 0)
            images[i, 0, max(0,ry-2):ry+2, max(0,rx-2):rx+2] = 1.0 
            
    # Save Grid
    save_path = os.path.join(CONFIG["RESULTS_DIR"], f"policy_e{epoch}.png")
    save_image(images, save_path, nrow=4)
    print(f"   >>> [Viz] Saved target practice: {save_path}")
    policy.train()

# --- TRAINING LOOP ---
def train():
    # Load Models
    policy = TitanPolicy(latent_dim=CONFIG["LATENT_DIM"]).to(device)
    optimizer = optim.AdamW(policy.parameters(), lr=CONFIG["LR"], weight_decay=1e-5)
    scaler = torch.amp.GradScaler('cuda') 
    
    # Load VAE (Frozen) for visualization
    vae = VAE(latent_dim=CONFIG["LATENT_DIM"]).to(device)
    try:
        vae.load_state_dict(torch.load(os.path.join(CONFIG["CHECKPOINT_DIR"], "vae_latest.pth"))['model_state_dict'])
        vae.requires_grad_(False)
        print(">>> VAE loaded for visualization.")
    except:
        print("!!! WARNING: VAE not found. Visualization will fail.")
        vae = None

    # Load Policy Checkpoint
    chk_path = os.path.join(CONFIG["CHECKPOINT_DIR"], CONFIG["SAVE_NAME"])
    if os.path.exists(chk_path):
        try:
            policy.load_state_dict(torch.load(chk_path))
            print(f">>> Resumed Policy from {chk_path}")
        except: pass

    dataset = PolicyDataset()
    if len(dataset) == 0: return

    loader = DataLoader(dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=True, num_workers=4)
    
    mse = nn.MSELoss()
    bce_logits = nn.BCEWithLogitsLoss() 

    print(">>> STARTING POLICY TRAINING...")
    
    for epoch in range(CONFIG["EPOCHS"]):
        policy.train()
        total_move_error = 0
        
        for i, (state, target) in enumerate(loader):
            state, target = state.to(device), target.to(device)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                pred = policy(state)
                
                # Split
                pred_xy = pred[:, :2] # Sigmoid (0-1)
                target_xy = target[:, :2]
                
                pred_aux = pred[:, 2:] # Logits
                target_aux = target[:, 2:]
                
                # Loss
                loss_move = mse(pred_xy, target_xy) * 10.0
                loss_aux = bce_logits(pred_aux, target_aux)
                loss = loss_move + loss_aux
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_move_error += loss_move.item()
            
            if i % 100 == 0:
                print(f"   [Ep {epoch} | B {i}] Move MSE: {loss_move.item():.4f}")

        # Save & Visualize
        torch.save(policy.state_dict(), chk_path)
        
        # Visualize every 2 epochs
        if epoch % 2 == 0 and vae is not None:
            save_target_practice(policy, vae, dataset, epoch)

if __name__ == "__main__":
    train()