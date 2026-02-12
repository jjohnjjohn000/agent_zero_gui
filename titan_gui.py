import sys
import os
import time

# --- CRITICAL: Environment Setup ---
# Prevents libraries (NumPy/Torch) from fighting over threads causing segfaults
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['PYTORCH_HIP_ALLOC_CONF'] = 'expandable_segments:True'

import multiprocessing

# Force spawn method for safety (Linux specific)
if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import queue

# --- MATPLOTLIB CONFIGURATION ---
import matplotlib
# We set the backend, but we DO NOT import the canvas widget yet.
# Importing 'backend_tkagg' before 'tk.Tk()' creates the crash.
matplotlib.use("TkAgg")
from matplotlib.figure import Figure

# =============================================================================
# GUI MAIN CLASS
# =============================================================================
class TitanGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("TITAN V4.0 - PERCEPTUAL VAE (Stable)")
        self.root.geometry("1450x900")
        
        # Graceful Exit
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # State
        self.process = None
        self.status_queue = None
        self.stop_event = None
        self.loss_history = []
        
        # Layout Division
        self.main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.main_paned.pack(fill=tk.BOTH, expand=True)
        
        self.left_panel = ttk.Frame(self.main_paned, width=380, relief=tk.RAISED)
        self.right_panel = ttk.Frame(self.main_paned)
        self.main_paned.add(self.left_panel, weight=1)
        self.main_paned.add(self.right_panel, weight=4)
        
        self.build_left_panel()
        self.build_right_panel()
        
        print(">>> TITAN GUI INITIALIZED.")
        self.root.after(500, self.poll_status)

    def build_left_panel(self):
        pad = {'padx': 10, 'pady': 5}
        tk.Label(self.left_panel, text="👁 TITAN PERCEPTUAL", font=("Helvetica", 16, "bold"), fg="#8e44ad").pack(pady=15)
        
        f_act = ttk.LabelFrame(self.left_panel, text="Actions")
        f_act.pack(fill="x", **pad)
        
        self.btn_start = tk.Button(f_act, text="▶ START NEW RUN", bg="#2ecc71", fg="black", font=("Arial", 11, "bold"), command=self.start_training)
        self.btn_start.pack(fill="x", padx=5, pady=5)
        
        self.btn_stop = tk.Button(f_act, text="⏹ STOP ENGINE", bg="#e74c3c", fg="white", font=("Arial", 11, "bold"), command=self.stop_training, state="disabled")
        self.btn_stop.pack(fill="x", padx=5, pady=5)
        
        # Hyperparameters
        f_hyp = ttk.LabelFrame(self.left_panel, text="Hyperparameters")
        f_hyp.pack(fill="x", **pad)
        
        self.vars = {}
        defaults = {
            "LEARNING_RATE": 0.00005, "BATCH_SIZE": 4, "LATENT_DIM": 4096,
            "EPOCHS": 150, "PERCEPTUAL_WEIGHT": 2.0, "CHROMA_WEIGHT": 30.0, "BETA": 0.00001
        }
        
        row = 0
        for k, v in defaults.items():
            ttk.Label(f_hyp, text=k.replace("_", " ").title()).grid(row=row, column=0, sticky="w", padx=5)
            if isinstance(v, float):
                var = tk.DoubleVar(value=v)
                sp = tk.Spinbox(f_hyp, textvariable=var, from_=0.0, to=100000.0, increment=0.00001 if v < 1 else 1, width=12)
            else:
                var = tk.IntVar(value=v)
                sp = tk.Spinbox(f_hyp, textvariable=var, from_=0, to=100000, increment=1, width=12)
            sp.grid(row=row, column=1, sticky="e", padx=5, pady=2)
            self.vars[k] = var
            row += 1
            
        # Path
        f_path = ttk.LabelFrame(self.left_panel, text="Dataset Path")
        f_path.pack(fill="x", **pad)
        self.path_var = tk.StringVar(value=os.path.abspath("./data"))
        tk.Entry(f_path, textvariable=self.path_var).pack(side=tk.LEFT, fill="x", expand=True, padx=2)
        tk.Button(f_path, text="...", command=self.browse_path).pack(side=tk.RIGHT, padx=2)

    def build_right_panel(self):
        self.notebook = ttk.Notebook(self.right_panel)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        self.tab_metrics = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_metrics, text="📊 Live Metrics")
        
        self.status_frame = ttk.LabelFrame(self.tab_metrics, text="Engine Status")
        self.status_frame.pack(fill="x", padx=10, pady=5)
        
        self.lbl_status_main = tk.Label(self.status_frame, text="READY", font=("Consolas", 14, "bold"), fg="gray")
        self.lbl_status_main.pack(pady=5)
        self.lbl_details = tk.Label(self.status_frame, text="Waiting for start...", font=("Arial", 10))
        self.lbl_details.pack(pady=2)
        
        # Graph
        self.fig = Figure(figsize=(6, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("VGG Perceptual Loss")
        self.ax.grid(True, alpha=0.3)
        self.line, = self.ax.plot([], [], 'm-', linewidth=1.5) # Magenta for Perceptual
        
        # --- SAFE IMPORT: Import Backend ONLY when needed ---
        try:
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            self.canvas = FigureCanvasTkAgg(self.fig, master=self.tab_metrics)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        except ImportError as e:
            tk.Label(self.tab_metrics, text=f"Graph Error: {e}", fg="red").pack()
        
        self.tab_logs = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_logs, text="📝 Engine Logs")
        self.log_text = scrolledtext.ScrolledText(self.tab_logs, state='disabled', font=("Consolas", 9))
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def browse_path(self):
        d = filedialog.askdirectory()
        if d: self.path_var.set(d)

    def write_log(self, msg):
        self.log_text.config(state='normal')
        self.log_text.insert('end', msg)
        self.log_text.see('end')
        self.log_text.config(state='disabled')

    def start_training(self):
        if self.process and self.process.is_alive(): return
        
        print(">>> STARTING TRAINING SUBPROCESS...")
        config = {k: v.get() for k, v in self.vars.items()}
        config["DATA_DIR"] = self.path_var.get()
        
        self.status_queue = multiprocessing.Queue()
        self.stop_event = multiprocessing.Event()
        
        # --- SAFE LAUNCH ---
        try:
            import train_vision_robust as engine
            self.process = multiprocessing.Process(
                target=engine.run_process_safe_entry, # Calls function in LOGIC file
                args=(self.status_queue, self.stop_event, config)
            )
            self.process.start()
            
            self.btn_start.config(state="disabled")
            self.btn_stop.config(state="normal")
            self.lbl_status_main.config(text="STARTING...", fg="#8e44ad")
            self.loss_history = []
            self.line.set_data([], [])
            self.canvas.draw()
        except Exception as e:
            messagebox.showerror("Launch Error", f"Failed to start engine: {e}")

    def stop_training(self):
        if self.process and self.process.is_alive():
            self.stop_event.set()
            self.lbl_status_main.config(text="STOPPING...", fg="red")
            self.btn_stop.config(state="disabled")

    def poll_status(self):
        if self.status_queue:
            try:
                while True:
                    data = self.status_queue.get_nowait()
                    msg_type = data.get("status")
                    if msg_type == "LOG":
                        self.write_log(data.get("msg") + "\n")
                    elif msg_type == "TRAINING":
                        loss = data.get("loss")
                        self.lbl_status_main.config(text=f"LOSS: {loss:.4f}", fg="blue")
                        
                        # --- V4 UPDATE: SHOW PERCEPTUAL WEIGHT ---
                        perc_val = data.get('perc', 0.0)
                        self.lbl_details.config(text=f"Epoch: {data.get('epoch')} | PercW: {perc_val:.2f}")
                        
                        # --- SYNC HYPERPARAMETERS WITH LIVE ENGINE ---
                        if "lr" in data: 
                            self.vars["LEARNING_RATE"].set(data["lr"])
                        # Check for 'perc' instead of 'edge_weight'
                        if "perc" in data: 
                            self.vars["PERCEPTUAL_WEIGHT"].set(data["perc"])
                        if "beta" in data: 
                            self.vars["BETA"].set(data["beta"])
                        # ---------------------------------------------

                        self.loss_history.append(loss)
                        if len(self.loss_history) % 10 == 0:
                            self.line.set_data(range(len(self.loss_history)), self.loss_history)
                            self.ax.relim()
                            self.ax.autoscale_view()
                            self.canvas.draw_idle()
                    elif msg_type == "STOPPED":
                        self.lbl_status_main.config(text="STOPPED", fg="gray")
                        self.btn_start.config(state="normal")
                        self.btn_stop.config(state="disabled")
                    elif msg_type == "ERROR":
                        self.write_log(f"ERROR: {data.get('msg')}\n")
                        self.lbl_status_main.config(text="ERROR", fg="red")
            except queue.Empty:
                pass
        
        if self.process and not self.process.is_alive() and self.btn_start['state'] == 'disabled':
            self.btn_start.config(state="normal")
            self.btn_stop.config(state="disabled")
            self.lbl_status_main.config(text="CRASHED/EXITED", fg="red")

        self.root.after(100, self.poll_status)

    def on_close(self):
        if self.process and self.process.is_alive():
            self.stop_event.set()
            self.process.join(timeout=1)
            if self.process.is_alive(): self.process.terminate()
        self.root.destroy()
        sys.exit(0)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    print(">>> DEBUG: Initializing Tk Root Window...")
    
    # 1. Create Root Window FIRST
    root = tk.Tk()
    
    # 2. Initialize Application (Imports Matplotlib backend internally)
    app = TitanGUI(root)
    
    # 3. Start Mainloop
    root.mainloop()