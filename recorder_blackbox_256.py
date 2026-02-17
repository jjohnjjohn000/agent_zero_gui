import time
import os
import threading
import queue
import numpy as np
import cv2
import mss
from pynput import mouse, keyboard
from datetime import datetime

# --- CONFIG ---
DATA_DIR = "data_actions"  # New folder for Action-Labeled Data
FPS = 4.0
CHUNK_SIZE = 500           # Save file every 500 frames
RESIZE_W, RESIZE_H = 256, 256

os.makedirs(DATA_DIR, exist_ok=True)

class input_listener:
    def __init__(self):
        self.mouse_x = 0
        self.mouse_y = 0
        self.mouse_click = 0 # 0=None, 1=Left, 2=Right
        self.keys_pressed = set()
        self.lock = threading.Lock()
        
        # Listeners
        self.m_listener = mouse.Listener(on_move=self.on_move, on_click=self.on_click)
        self.k_listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        
        self.m_listener.start()
        self.k_listener.start()

    def on_move(self, x, y):
        with self.lock:
            self.mouse_x = x
            self.mouse_y = y

    def on_click(self, x, y, button, pressed):
        with self.lock:
            if pressed:
                if button == mouse.Button.left: self.mouse_click = 1
                elif button == mouse.Button.right: self.mouse_click = 2
            else:
                self.mouse_click = 0

    def on_press(self, key):
        with self.lock:
            try: k = key.char # Letters
            except: k = str(key) # Special keys
            self.keys_pressed.add(k)

    def on_release(self, key):
        with self.lock:
            try: k = key.char
            except: k = str(key)
            if k in self.keys_pressed: self.keys_pressed.remove(k)

    def get_state(self):
        with self.lock:
            # Simple encoding: [x, y, click, num_keys_pressed]
            # Ideally we'd encode specific keys, but for now we track activity density
            return [self.mouse_x, self.mouse_y, self.mouse_click, len(self.keys_pressed)]

def record_loop():
    print(f">>> RECORDER V2 STARTED. Saving to '{DATA_DIR}'...")
    print(">>> Press Ctrl+C in terminal to stop.")
    
    inputs = input_listener()
    sct = mss.mss()
    
    buffer_imgs = []
    buffer_acts = []
    
    frame_time = 1.0 / FPS
    
    try:
        while True:
            start_t = time.time()
            
            # 1. Grab Screen
            monitor = sct.monitors[1] # Main monitor
            img = np.array(sct.grab(monitor))
            
            # 2. Grab Input State (synchronized)
            action = inputs.get_state() # [x, y, click, keys]
            
            # 3. Process Image (Resize & RGB)
            # Remove alpha channel, Convert BGR -> RGB, Resize
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            img = cv2.resize(img, (RESIZE_W, RESIZE_H), interpolation=cv2.INTER_AREA)
            
            # 4. Store
            buffer_imgs.append(img)
            buffer_acts.append(action)
            
            # 5. Flush if full
            if len(buffer_imgs) >= CHUNK_SIZE:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                fname = os.path.join(DATA_DIR, f"rec_{ts}.npz")
                
                # Save compressed
                np.savez_compressed(
                    fname, 
                    images=np.array(buffer_imgs, dtype=np.uint8),
                    actions=np.array(buffer_acts, dtype=np.int32)
                )
                
                print(f" [Saved] {fname} ({len(buffer_imgs)} frames)")
                buffer_imgs = []
                buffer_acts = []
                
            # Sleep to maintain FPS
            elapsed = time.time() - start_t
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)
                
    except KeyboardInterrupt:
        print("\n>>> STOPPING...")
        if len(buffer_imgs) > 0:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = os.path.join(DATA_DIR, f"rec_{ts}_final.npz")
            np.savez_compressed(fname, images=np.array(buffer_imgs), actions=np.array(buffer_acts))
            print(f" [Saved Final] {fname}")

if __name__ == "__main__":
    record_loop()