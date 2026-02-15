import pyautogui
import time
import sys

print(">>> MOVE YOUR MOUSE AROUND! (Press Ctrl+C to stop)")
try:
    while True:
        x, y = pyautogui.position()
        print(f"\rCurrent Position: X={x}, Y={y}      ", end="")
        sys.stdout.flush()
        time.sleep(0.1)
except KeyboardInterrupt:
    print("\nDone.")