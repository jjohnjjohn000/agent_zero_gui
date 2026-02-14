import time
import cv2
import mss
import numpy as np
import torch
import pyautogui
import sys
import os
import threading
from pynput import keyboard

# Imports
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
UPDATE_RATE = 3        # Reduced to 5Hz (200ms between updates) for smoother movement
MOVE_SPEED = 0.15      # Increased to 150ms (must be < 200ms for smooth completion)
MOVE_THRESHOLD = 0.05  # Increased to 3% to ignore micro-jitters
EMA_ALPHA = 0.2        # Exponential moving average weight for smoothing (0.0=all history, 1.0=no smoothing)

# Safety
pyautogui.FAILSAFE = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AgentZero:
    def __init__(self):
        print(">>> INITIALIZING AGENT ZERO...")
        
        self.kill_switch = False
        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()
        
        self.vae = VAE(latent_dim=LATENT_DIM, img_size=TARGET_SIZE).to(device)
        self.load_weights(self.vae, "vae_latest.pth")
        self.vae.eval()
        
        self.policy = TitanPolicy(latent_dim=LATENT_DIM).to(device)
        self.load_weights(self.policy, "policy_latest.pth")
        self.policy.eval()
        
        # Initialize Vision State
        self.sct = mss.mss()
        self.monitor = self.get_target_monitor()
        self.prev_img = None 
        self.last_target_x = None
        self.last_target_y = None
        
        # EMA smoothing for stable predictions
        self.ema_x = 0.5  # Start at screen center
        self.ema_y = 0.5
        
        print(f">>> LOCKED MONITOR: {self.monitor}")
        print(">>> STARTING IN 3 SECONDS...")
        time.sleep(3)

    def get_target_monitor(self):
        mx, my = pyautogui.position()
        monitors = self.sct.monitors[1:] 
        for i, m in enumerate(monitors):
            if (m['left'] <= mx < m['left'] + m['width']) and \
               (m['top'] <= my < m['top'] + m['height']):
                return m
        return monitors[0]

    def on_press(self, key):
        if key == keyboard.Key.esc:
            print("\n>>> ESC PRESSED.")
            self.kill_switch = True
            os._exit(0)

    def load_weights(self, model, filename):
        path = f"{CHECKPOINT_DIR}/{filename}"
        if not os.path.exists(path):
            print(f"!!! MISSING: {path}")
            sys.exit()
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        state = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        model.load_state_dict(state)

    def see(self, frame_count):
        # Capture
        sct_img = self.sct.grab(self.monitor)
        img = np.array(sct_img)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        
        # Resize
        img_small = cv2.resize(img, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA)
        
        # --- VISION PULSE CHECK ---
        diff_score = 0.0
        if self.prev_img is not None:
            # Calculate absolute difference between frames
            diff = cv2.absdiff(img_small, self.prev_img)
            diff_score = np.mean(diff)
        
        self.prev_img = img_small
        
        # Save debug view more frequently to diagnose jitter
        if frame_count % 5 == 0:
            cv2.imwrite("agent_view.jpg", cv2.cvtColor(img_small, cv2.COLOR_RGB2BGR))
        
        # Process for AI
        tensor = torch.from_numpy(img_small).float().permute(2, 0, 1) / 255.0
        tensor = tensor.unsqueeze(0).to(device)
        
        with torch.no_grad():
            mu, _ = self.vae.encode(tensor)
            
        return mu, diff_score

    def act(self, latent, diff_score):
        if self.kill_switch: return

        with torch.no_grad():
            action_vec = self.policy(latent)
            coords = action_vec[0, :2]
        
        # Policy already outputs [0, 1] via sigmoid in head_mouse
        x_raw = coords[0].item()
        y_raw = coords[1].item()
        
        # Apply Exponential Moving Average for smooth movement
        self.ema_x = EMA_ALPHA * x_raw + (1 - EMA_ALPHA) * self.ema_x
        self.ema_y = EMA_ALPHA * y_raw + (1 - EMA_ALPHA) * self.ema_y
        
        x_norm = self.ema_x
        y_norm = self.ema_y
        
        # Calculate target pixel coordinates
        target_x = int(self.monitor['left'] + (x_norm * self.monitor['width']))
        target_y = int(self.monitor['top']  + (y_norm * self.monitor['height']))
        
        # Clamp to monitor bounds
        target_x = max(self.monitor['left'], min(self.monitor['left'] + self.monitor['width'] - 2, target_x))
        target_y = max(self.monitor['top'],  min(self.monitor['top']  + self.monitor['height'] - 2, target_y))
        
        # Check if we should move (distance threshold)
        should_move = True
        distance = 0.0
        if self.last_target_x is not None:
            dx = (target_x - self.last_target_x) / self.monitor['width']
            dy = (target_y - self.last_target_y) / self.monitor['height']
            distance = np.sqrt(dx**2 + dy**2)
            should_move = distance > MOVE_THRESHOLD
        
        # PRINT STATUS
        status = "FROZEN ❄️" if diff_score < 0.1 else "LIVE 🟢"
        move_status = "→MOVE" if should_move else "  HOLD"
        sys.stdout.write(f"\r>>> VIS: {status} ({diff_score:.2f}) | RAW: ({x_raw:.3f}, {y_raw:.3f}) | SMOOTH: ({x_norm:.3f}, {y_norm:.3f}) | DIST: {distance:.4f} {move_status}   ")
        sys.stdout.flush()

        # Execute movement
        if should_move:
            try:
                pyautogui.moveTo(target_x, target_y, duration=MOVE_SPEED, tween=pyautogui.easeOutQuad)
                self.last_target_x = target_x
                self.last_target_y = target_y
            except pyautogui.FailSafeException:
                print("\n>>> FAILSAFE.")
                os._exit(0)

    def run(self):
        print(">>> AGENT ACTIVE.")
        frame_count = 0
        try:
            while not self.kill_switch:
                start = time.time()
                
                thought, diff = self.see(frame_count)
                self.act(thought, diff)
                
                frame_count += 1
                elapsed = time.time() - start
                wait = (1.0 / UPDATE_RATE) - elapsed
                if wait > 0: time.sleep(wait)
                    
        except KeyboardInterrupt:
            print("\n>>> STOPPED.")

if __name__ == "__main__":
    bot = AgentZero()
    bot.run()