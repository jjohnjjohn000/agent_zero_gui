import numpy as np
import glob
import os

files = sorted(glob.glob("./data/memories/*.npz"))
print(f"Found {len(files)} memory files.")

print("\n--- CHECKING FOR LIFE SIGNS ---")
active_files = 0
for i, f in enumerate(files[:275]): # Check first 50
    try:
        data = np.load(f)
        actions = data['actions'] # Shape (T, 6)
        
        # Calculate movement (Standard Deviation of X and Y)
        # If std is 0, the mouse never moved.
        move_x = np.std(actions[:, 0])
        move_y = np.std(actions[:, 1])
        
        is_active = (move_x > 0.001) or (move_y > 0.001)
        
        if is_active:
            active_files += 1
            print(f"File {i}: ✅ ACTIVE | Range X: {actions[:,0].min():.2f}-{actions[:,0].max():.2f}")
        else:
            print(f"File {i}: 💤 STATIC (Mouse stuck at {actions[0,0]:.2f}, {actions[0,1]:.2f})")
            
    except: pass

print(f"\nSummary: {active_files} active files found in the first 50.")