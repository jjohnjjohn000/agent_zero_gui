import time
import cv2
import mss
import numpy as np
import torch
import pyautogui
import sys
import os
from pynput import keyboard

try:
    from models.vae import VAE
    from models.policy import TitanPolicy
except ImportError:
    sys.path.append(os.getcwd())
    from models.vae import VAE
    from models.policy import TitanPolicy

# --- CONFIG ---
CHECKPOINT_DIR = "./checkpoints"
TARGET_SIZE = 256
LATENT_DIM = 4096
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AgentDebugger:
    def __init__(self):
        print(">>> INITIALIZING DEBUGGER...")
        
        # Load Models
        self.vae = VAE(latent_dim=LATENT_DIM, img_size=TARGET_SIZE).to(device)
        self.load_weights(self.vae, "vae_latest.pth")
        self.vae.eval()
        
        self.policy = TitanPolicy(latent_dim=LATENT_DIM).to(device)
        self.load_weights(self.policy, "policy_latest.pth")
        self.policy.eval()
        
        self.sct = mss.mss()
        self.monitor = self.sct.monitors[1] # Default to 1
        
        print(">>> DEBUG MODE STARTED. MOVING MOUSE TO DETECT MONITOR...")
        self.monitor = self.get_current_monitor()
        print(f">>> Target Monitor: {self.monitor}")

    def get_current_monitor(self):
        mx, my = pyautogui.position()
        monitors = self.sct.monitors[1:]
        for i, m in enumerate(monitors):
            if (m['left'] <= mx < m['left'] + m['width']) and \
               (m['top'] <= my < m['top'] + m['height']):
                return m
        return monitors[0]

    def load_weights(self, model, filename):
        path = f"{CHECKPOINT_DIR}/{filename}"
        if not os.path.exists(path):
            print(f"!!! MISSING: {path}")
            sys.exit()
        checkpoint = torch.load(path, map_location=device)
        state = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        model.load_state_dict(state)

    def run(self):
        print("\n>>> STARTING DIAGNOSTIC LOOP (Press Ctrl+C to stop)")
        try:
            while True:
                # 1. CAPTURE
                sct_img = self.sct.grab(self.monitor)
                img = np.array(sct_img)
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
                
                # CHECK 1: IS IMAGE EMPTY?
                avg_color = img.mean()
                
                # 2. ENCODE
                img_small = cv2.resize(img, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA)
                tensor = torch.from_numpy(img_small).float().permute(2, 0, 1) / 255.0
                tensor = tensor.unsqueeze(0).to(device)
                
                with torch.no_grad():
                    mu, _ = self.vae.encode(tensor)
                    
                    # CHECK 2: IS LATENT DEAD?
                    latent_std = mu.std().item()
                    
                    # 3. ACT
                    action = self.policy(mu)
                    raw_x = action[0, 0].item()
                    raw_y = action[0, 1].item()
                
                # REPORT
                status = "✅ OK" if avg_color > 10 else "❌ BLIND"
                print(f"\rStatus: {status} | Brightness: {avg_color:.1f} | Brain Activity: {latent_std:.4f} | Want to go: ({raw_x:.2f}, {raw_y:.2f})    ", end="")
                
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\nDone.")

if __name__ == "__main__":
    bot = AgentDebugger()
    bot.run()