import numpy as np
import glob
import os

def check():
    # 1. Check Raw Recordings
    raw_files = sorted(glob.glob("data/*.npz"))
    print(f"--- RAW DATA DIAGNOSTIC ({len(raw_files)} files) ---")
    
    if not raw_files:
        print("❌ No raw recordings found!")
    else:
        sizes = []
        for f in raw_files[:10]: # Check first 10
            try:
                with np.load(f) as data:
                    n = len(data['images'])
                    sizes.append(n)
                    print(f"File {os.path.basename(f)}: {n} frames")
            except Exception as e:
                print(f"❌ Corrupt file {f}: {e}")
        
        avg = sum(sizes) / len(sizes) if sizes else 0
        print(f"👉 Average Length: {avg:.1f} frames")
        if avg < 65:
            print("⚠️ WARNING: Your recordings are too short! The Brain needs sequences > 64 frames.")

    # 2. Check Compiled Memories
    mem_files = sorted(glob.glob("data/memories/*.npz"))
    print(f"\n--- MEMORY DIAGNOSTIC ({len(mem_files)} files) ---")
    
    if not mem_files:
        print("❌ No memories found. You need to run compile_memories.py!")
    else:
        valid_count = 0
        for f in mem_files[:10]:
            try:
                with np.load(f) as data:
                    n = len(data['latents'])
                    if n > 64: valid_count += 1
                    print(f"Memory {os.path.basename(f)}: {n} steps")
            except:
                print(f"❌ Corrupt memory {f}")

        if valid_count == 0:
            print("\n❌ CRITICAL: No valid memories found for training (Length > 64).")
            print("SOLUTION: Delete the 'data/memories' folder and re-run compile_memories.py")
        else:
            print("\n✅ Data looks good. You should be able to train.")

if __name__ == "__main__":
    check()