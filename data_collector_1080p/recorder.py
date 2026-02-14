# data_collector/recorder.py
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

import time
import cv2
import mss
import numpy as np
import threading
from pynput import mouse, keyboard
from config import FPS, RESIZE_FACTOR, CHUNK_SIZE
from io_utils import save_chunk

class DataRecorder:
    def __init__(self):
        self.running = False
        self.mouse_events = []
        self.keyboard_events = []
        
        # Buffer
        self.frame_buffer = []
        self.mouse_buffer = []
        self.key_buffer = []
        self.meta_buffer = [] # Stores which monitor is active
        
        # Screen capture setup
        self.sct = mss.mss()
        self.monitors = self.sct.monitors[1:] # Skip [0] (All monitors combined)
        
        # Mouse controller to poll position instantly
        self.mouse_controller = mouse.Controller()
        
        # Thread locks
        self.lock = threading.Lock()

    def get_active_monitor(self):
        """
        Returns the monitor dict (mss format) that contains the mouse.
        Also returns the monitor index (0 for left, 1 for right, etc).
        """
        x, y = self.mouse_controller.position
        
        for i, monitor in enumerate(self.monitors):
            # Check if mouse is within this monitor's bounding box
            if (monitor["left"] <= x < monitor["left"] + monitor["width"] and
                monitor["top"] <= y < monitor["top"] + monitor["height"]):
                return monitor, i
        
        # Fallback: If mouse is somehow off-screen, return primary (usually index 0)
        return self.monitors[0], 0

    def on_move(self, x, y):
        with self.lock:
            self.mouse_events.append({'t': time.time(), 'action': 'move', 'x': x, 'y': y})

    def on_click(self, x, y, button, pressed):
        with self.lock:
            action = 'pressed' if pressed else 'released'
            self.mouse_events.append({
                't': time.time(), 
                'action': 'click', 
                'x': x, 
                'y': y, 
                'button': str(button),
                'state': action
            })

    def on_press(self, key):
        with self.lock:
            try:
                k = key.char
            except AttributeError:
                k = str(key)
            self.keyboard_events.append({'t': time.time(), 'action': 'press', 'key': k})

    def capture_cycle(self):
        print(f"Recorder started. Detecting {len(self.monitors)} monitors.")
        print(f"Targeting active monitor based on mouse position.")
        print("Press Ctrl+C to stop.")
        
        try:
            while self.running:
                start_time = time.time()

                # 1. Determine which monitor to record
                active_mon, mon_index = self.get_active_monitor()

                # 2. Grab Screen of THAT monitor only
                sct_img = self.sct.grab(active_mon)
                img = np.array(sct_img)
                
                # BGRA -> BGR
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

                # Resize
                if RESIZE_FACTOR != 1.0:
                    width = int(img.shape[1] * RESIZE_FACTOR)
                    height = int(img.shape[0] * RESIZE_FACTOR)
                    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

                # 3. Grab Inputs
                with self.lock:
                    current_mouse = self.mouse_events[:]
                    current_key = self.keyboard_events[:]
                    self.mouse_events = []
                    self.keyboard_events = []

                # 4. Add to Buffer
                self.frame_buffer.append(img)
                self.mouse_buffer.append(current_mouse)
                self.key_buffer.append(current_key)
                self.meta_buffer.append({'monitor_idx': mon_index, 'w': active_mon['width'], 'h': active_mon['height']})

                # 5. Save if full
                if len(self.frame_buffer) >= CHUNK_SIZE:
                    threading.Thread(target=save_chunk, args=(
                        self.frame_buffer[:], 
                        self.mouse_buffer[:], 
                        self.key_buffer[:],
                        self.meta_buffer[:] # Pass metadata too
                    )).start()
                    
                    self.frame_buffer = []
                    self.mouse_buffer = []
                    self.key_buffer = []
                    self.meta_buffer = []

                # 6. Maintain FPS
                elapsed = time.time() - start_time
                sleep_time = (1.0 / FPS) - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("Stopping...")
        finally:
            if self.frame_buffer:
                print("Saving remaining data...")
                # Note: You need to update io_utils.py to accept the 4th argument (meta)
                # For now, we just pass the first 3 to avoid breaking if you didn't update io_utils yet.
                # ideally, update save_chunk to accept **kwargs
                save_chunk(self.frame_buffer, self.mouse_buffer, self.key_buffer, self.meta_buffer)

    def start(self):
        self.running = True
        
        # Listeners (Non-blocking)
        m_listener = mouse.Listener(on_move=self.on_move, on_click=self.on_click)
        k_listener = keyboard.Listener(on_press=self.on_press)
        
        m_listener.start()
        k_listener.start()

        self.capture_cycle()

        m_listener.stop()
        k_listener.stop()

if __name__ == "__main__":
    rec = DataRecorder()
    rec.start()