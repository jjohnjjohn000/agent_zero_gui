import os
import sys
import time
import queue
import threading
import collections
import numpy as np
import sounddevice as sd
import torch
from faster_whisper import WhisperModel
from colorama import init, Fore, Style

# Initialize colorama for HUD-like terminal output
init(autoreset=True)

# ==================================================================================
# CONFIGURATION - TUNED FOR TECHNICAL DICTATION
# ==================================================================================
class AudioConfig:
    # DEVICE
    DEVICE_INDEX = None        # Set to integer index if you have multiple mics, else None for default
    SAMPLE_RATE = 16000        # Whisper expects 16kHz
    BLOCK_SIZE = 512           # Buffer chunk size (latency vs cpu load)
    
    # VAD (Voice Activity Detection)
    VAD_THRESHOLD = 0.5        # Confidence threshold (0.1 - 0.9)
    MIN_SPEECH_DURATION = 0.2  # Ignore chirps shorter than 200ms
    SILENCE_TIMEOUT = 0.8      # How long to wait after speech stops to commit the sentence
    PRE_ROLL_DURATION = 0.5    # How much audio to keep *before* the VAD trigger (seconds)
    
    # WHISPER
    MODEL_SIZE = "large-v3"    # The big gun. Use "medium.en" if VRAM is tight (<6GB)
    COMPUTE_TYPE = "float16"   # "float16" or "int8_float16"
    BEAM_SIZE = 5              # Accuracy vs Speed
    
    # PROMPTING (The secret sauce for accuracy)
    # This forces the model to expect code and system commands, not conversational English.
    INITIAL_PROMPT = (
        "Titan system command interface. Execute python script. "
        "Sudo apt install. Git push origin master. Docker run. "
        "Initialize neural network. 192.168.1.1. Hash function."
    )

# ==================================================================================
# ROBUST TITAN EAR CLASS
# ==================================================================================
class TitanEar:
    def __init__(self, on_speech_detected_callback=None):
        """
        :param on_speech_detected_callback: Function to call with text when transcription finishes.
        """
        self.running = False
        self.callback = on_speech_detected_callback
        
        # 1. Thread-Safe Queues
        self.raw_queue = queue.Queue()          # Audio from Mic -> VAD
        self.transcribe_queue = queue.Queue()   # Audio from VAD -> Whisper

        # 2. Load VAD (CPU - Instant)
        print(f"{Fore.CYAN}[SYSTEM] Loading Silero VAD Gatekeeper...{Style.RESET_ALL}")
        try:
            self.vad_model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=True, # ONNX is faster for CPU inference
                trust_repo=True
            )
            self.get_speech_timestamps, _, _, _, _ = utils
        except Exception as e:
            print(f"{Fore.RED}[ERROR] VAD Load Failed: {e}{Style.RESET_ALL}")
            sys.exit(1)

        # 3. Load Whisper (GPU - Heavy)
        print(f"{Fore.CYAN}[SYSTEM] Loading Whisper {AudioConfig.MODEL_SIZE} on GPU...{Style.RESET_ALL}")
        try:
            self.model = WhisperModel(
                AudioConfig.MODEL_SIZE, 
                device="cuda", 
                compute_type=AudioConfig.COMPUTE_TYPE
            )
        except Exception as e:
            print(f"{Fore.RED}[CRITICAL] Whisper Load Failed. Is CUDA available? {e}{Style.RESET_ALL}")
            sys.exit(1)

        print(f"{Fore.GREEN}>>> TITAN EAR INITIALIZED.{Style.RESET_ALL}")

    def start(self):
        """Ignites the engine threads."""
        self.running = True
        
        # Thread 1: Input (SoundDevice is its own thread usually, but we manage the callback)
        # Thread 2: VAD Processing
        self.vad_thread = threading.Thread(target=self._vad_worker, daemon=True)
        self.vad_thread.start()
        
        # Thread 3: Whisper Transcription
        self.transcribe_thread = threading.Thread(target=self._transcribe_worker, daemon=True)
        self.transcribe_thread.start()
        
        # Start Microphone
        self.stream = sd.InputStream(
            device=AudioConfig.DEVICE_INDEX,
            samplerate=AudioConfig.SAMPLE_RATE,
            channels=1,
            callback=self._audio_callback,
            blocksize=AudioConfig.BLOCK_SIZE
        )
        self.stream.start()
        print(f"{Fore.GREEN}>>> LISTENING FOR COMMANDS...{Style.RESET_ALL}")

    def stop(self):
        """Graceful shutdown."""
        self.running = False
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        print(f"{Fore.YELLOW}[SYSTEM] Titan Ear Stopped.{Style.RESET_ALL}")

    def _audio_callback(self, indata, frames, time_info, status):
        """Hardware interrupt handler. Must be FAST."""
        if status:
            print(f"{Fore.RED}[AUDIO HARDWARE ERROR] {status}{Style.RESET_ALL}")
        # Copy data immediately to queue to release hardware buffer
        self.raw_queue.put(indata.copy())

    def _vad_worker(self):
        """
        The Gatekeeper. 
        Monitors the raw audio stream.
        Decides when a sentence starts and ends.
        """
        # Ring buffer for pre-roll (keeping the last 0.5s of audio)
        # Size = (SampleRate * Seconds) / BlockSize
        maxlen = int((AudioConfig.SAMPLE_RATE / AudioConfig.BLOCK_SIZE) * AudioConfig.PRE_ROLL_DURATION)
        ring_buffer = collections.deque(maxlen=maxlen)
        
        triggered = False
        voiced_frames = []
        silence_counter = 0
        
        # Calculations for timing
        chunks_per_second = AudioConfig.SAMPLE_RATE / AudioConfig.BLOCK_SIZE
        silence_threshold_chunks = AudioConfig.SILENCE_TIMEOUT * chunks_per_second

        while self.running:
            try:
                chunk = self.raw_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            # Flatten and normalize
            chunk = chunk.flatten().astype(np.float32)
            
            # --- VAD INFERENCE ---
            # Check if this specific chunk contains human speech
            # We wrap in try/except because Silero can be picky about tensor shapes
            try:
                chunk_tensor = torch.from_numpy(chunk)
                speech_prob = self.vad_model(chunk_tensor, AudioConfig.SAMPLE_RATE).item()
            except Exception:
                speech_prob = 0.0

            # --- STATE MACHINE ---
            current_is_speech = speech_prob > AudioConfig.VAD_THRESHOLD

            if not triggered:
                ring_buffer.append(chunk)
                if current_is_speech:
                    # START OF SENTENCE
                    print(f"{Fore.YELLOW}🎤 [RECORDING]...", end="\r", flush=True)
                    triggered = True
                    # Dump ring buffer (pre-roll) into voiced frames
                    voiced_frames.extend(ring_buffer)
                    voiced_frames.append(chunk)
                    ring_buffer.clear()
                    silence_counter = 0
            else:
                # WE ARE RECORDING
                voiced_frames.append(chunk)
                
                if current_is_speech:
                    silence_counter = 0
                else:
                    silence_counter += 1
                
                # END OF SENTENCE CHECK
                if silence_counter > silence_threshold_chunks:
                    triggered = False
                    
                    # Optimization: Don't transcribe if it was just a cough (< 0.2s)
                    total_duration = (len(voiced_frames) * AudioConfig.BLOCK_SIZE) / AudioConfig.SAMPLE_RATE
                    if total_duration > AudioConfig.MIN_SPEECH_DURATION:
                        # Send to Whisper
                        full_audio = np.concatenate(voiced_frames)
                        self.transcribe_queue.put(full_audio)
                        print(f"{Fore.MAGENTA}⚡ [PROCESSING]...   ", end="\r", flush=True)
                    else:
                        print(f"{Fore.BLACK}❌ [NOISE IGNORED]    ", end="\r", flush=True)
                    
                    voiced_frames = []
                    silence_counter = 0

    def _transcribe_worker(self):
        """
        The GPU Worker.
        Takes raw numpy arrays, runs the Heavy Transformer.
        """
        while self.running:
            try:
                audio_data = self.transcribe_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            try:
                # --- WHISPER INFERENCE ---
                segments, info = self.model.transcribe(
                    audio_data,
                    beam_size=AudioConfig.BEAM_SIZE,
                    language="en", # Hardcoded for speed and command accuracy
                    condition_on_previous_text=False, # Prevents hallucination loops
                    initial_prompt=AudioConfig.INITIAL_PROMPT, # Technical bias
                    vad_filter=True, # Double check
                    vad_parameters=dict(min_silence_duration_ms=500)
                )

                text = " ".join([segment.text for segment in segments]).strip()

                # --- VALIDATION ---
                # Whisper Large V3 sometimes hallucinates "Thank you." or subtitles in silence.
                # We filter out common hallucinations or empty strings.
                bad_phrases = ["Thank you.", "Thanks for watching", "You", "."]
                
                if text and text not in bad_phrases:
                    # SUCCESS
                    print(f"\n{Fore.GREEN}📝 [CMD]: {Style.BRIGHT}{text}{Style.RESET_ALL}")
                    
                    if self.callback:
                        self.callback(text)
                else:
                    # Hallucination or empty
                    print(f"{Fore.BLACK}❌ [NULL]               ", end="\r", flush=True)

            except Exception as e:
                print(f"\n{Fore.RED}[TRANSCRIPTION ERROR] {e}{Style.RESET_ALL}")

# ==================================================================================
# EXAMPLE USAGE / ENTRY POINT
# ==================================================================================
def main_agent_loop(text):
    """
    This is where you will hook in your Agent Logic later.
    For now, it just parses basic test commands.
    """
    cmd = text.lower()
    if "exit" in cmd or "stop" in cmd:
        print(f"{Fore.RED}>>> SHUTDOWN COMMAND RECEIVED.{Style.RESET_ALL}")
        return False # Signal to stop
    
    if "browser" in cmd:
        print(f"{Fore.BLUE}>>> [ACTION] Launching Browser...{Style.RESET_ALL}")
        # os.system("firefox") 
    
    return True

if __name__ == "__main__":
    # 1. Define the callback
    running = True
    def on_hear(text):
        global running
        should_continue = main_agent_loop(text)
        if not should_continue:
            running = False

    # 2. Start the Ear
    ear = TitanEar(on_speech_detected_callback=on_hear)
    ear.start()

    # 3. Main thread keeps alive (or runs GUI/Video loop)
    try:
        while running:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        ear.stop()