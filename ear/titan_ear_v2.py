#!/usr/bin/env python3
"""
TITAN EAR V2 - PRODUCTION SPEECH RECOGNITION SYSTEM
===================================================
Full-stack voice recognition with:
- Acoustic Echo Cancellation (AEC) - WORKING SPECTRAL SUBTRACTION
- Speaker Verification (your voice only)
- Voice Activity Detection (VAD)
- High-accuracy Whisper transcription
- French language support

Author: Advanced AI System
Version: 2.0 - Updated with Working AEC
"""

import os
import sys
import time
import queue
import threading
import collections
import numpy as np
import torch
from faster_whisper import WhisperModel
from colorama import init, Fore, Style

# Import our custom modules
try:
    from audio_capture_pulse import DualAudioCapturePulse as DualAudioCapture
    print(f"{Fore.GREEN}[IMPORT] Using PulseAudio native capture{Style.RESET_ALL}")
except ImportError:
    from audio_capture import DualAudioCapture
    print(f"{Fore.YELLOW}[IMPORT] Using standard audio capture{Style.RESET_ALL}")

from aec_processor import AECProcessor, SimpleNoiseGate
from speaker_verification import SpeakerVerifier, AdaptiveThreshold

# Initialize colorama
init(autoreset=True)

# Enable progress bars for downloads
os.environ['TQDM_DISABLE'] = '0'

# ==================================================================================
# CONFIGURATION - TUNED FOR FRENCH TECHNICAL DICTATION
# ==================================================================================
# QUICK ADJUSTMENTS:
# - AEC too aggressive? Decrease AEC_AGGRESSIVENESS to 2.0
# - AEC not enough? Increase AEC_AGGRESSIVENESS to 4.5
# - Speaker rejection too high? Decrease SPEAKER_THRESHOLD to 0.5
# - Want English instead? Change LANGUAGE="fr" to LANGUAGE="en"
# - Still missing speech? Decrease VAD_THRESHOLD to 0.65
# ==================================================================================

class AudioConfig:
    # DEVICE
    SAMPLE_RATE = 16000        # Whisper expects 16kHz
    BLOCK_SIZE = 512           # Buffer chunk size
    
    # AEC (Acoustic Echo Cancellation) - WORKING VERSION
    AEC_ENABLED = True         # Set False to disable AEC
    AEC_AGGRESSIVENESS = 3.0   # How much echo to remove (2.0-5.0, higher = more aggressive)
                               # 2.0 = gentle, 3.0 = balanced (default), 4.0 = aggressive, 5.0 = maximum
    
    # SPEAKER VERIFICATION
    SPEAKER_VERIFICATION_ENABLED = True  # Set False to disable
    SPEAKER_THRESHOLD = 0.2    # Similarity threshold (0.5=loose, 0.75=strict)
    SPEAKER_ADAPTIVE = True    # Auto-adjust threshold
    
    # VAD (Voice Activity Detection)
    VAD_THRESHOLD = 0.7        # Balanced for real voice (0.1 - 0.9)
    MIN_SPEECH_DURATION = 0.4  # Ignore chirps shorter than 400ms
    SILENCE_TIMEOUT = 1.2      # How long to wait after speech stops
    PRE_ROLL_DURATION = 0.5    # Audio kept before VAD trigger
    
    # WHISPER
    MODEL_SIZE = "large-v3"    # Use "medium.en" if RAM is tight
    COMPUTE_TYPE = "int8"      # CPU: int8, GPU: float16
    BEAM_SIZE = 5              # Accuracy vs Speed
    
    # LANGUAGE
    LANGUAGE = "fr"            # French (change to "en" for English)
    
    # PROMPTING
    INITIAL_PROMPT = (
        "Interface de commande système Titan. Exécuter le script python. "
        "Sudo apt install. Git push origin master. Docker run. "
        "Initialiser le réseau neuronal. Fonction de hachage."
    )

# ==================================================================================
# TITAN EAR V2 - MAIN CLASS
# ==================================================================================

class TitanEarV2:
    def __init__(self, on_speech_detected_callback=None):
        """
        Initialize Titan Ear v2 with full audio processing pipeline.
        
        :param on_speech_detected_callback: Function to call with transcribed text
        """
        self.running = False
        self.callback = on_speech_detected_callback
        
        # Queues
        self.raw_audio_queue = queue.Queue()
        self.verified_audio_queue = queue.Queue()
        self.transcribe_queue = queue.Queue()
        
        print(f"\n{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}       TITAN EAR V2 - INITIALIZATION{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}\n")
        
        # 1. Load VAD (Silero)
        print(f"{Fore.CYAN}[1/6] Loading Silero VAD...{Style.RESET_ALL}")
        try:
            self.vad_model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=True,
                trust_repo=True,
                verbose=False
            )
            self.get_speech_timestamps, _, _, _, _ = utils
            print(f"{Fore.GREEN}      ✓ VAD Ready{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}      ✗ VAD Failed: {e}{Style.RESET_ALL}")
            sys.exit(1)
        
        # 2. Initialize AEC (WORKING SPECTRAL SUBTRACTION)
        if AudioConfig.AEC_ENABLED:
            print(f"{Fore.CYAN}[2/6] Initializing AEC (Spectral Subtraction)...{Style.RESET_ALL}")
            self.aec = AECProcessor(
                sample_rate=AudioConfig.SAMPLE_RATE,
                aggressiveness=AudioConfig.AEC_AGGRESSIVENESS
            )
            self.noise_gate = SimpleNoiseGate(threshold_db=-35)
            print(f"{Fore.GREEN}      ✓ AEC Ready (aggressiveness={AudioConfig.AEC_AGGRESSIVENESS}){Style.RESET_ALL}")
        else:
            self.aec = None
            print(f"{Fore.YELLOW}[2/6] AEC Disabled{Style.RESET_ALL}")
        
        # 3. Load Speaker Verification
        if AudioConfig.SPEAKER_VERIFICATION_ENABLED:
            print(f"{Fore.CYAN}[3/6] Loading Speaker Verification...{Style.RESET_ALL}")
            try:
                self.speaker_verifier = SpeakerVerifier(
                    profile_path="user_voice_profile.pkl",
                    threshold=AudioConfig.SPEAKER_THRESHOLD
                )
                
                if AudioConfig.SPEAKER_ADAPTIVE:
                    self.adaptive_threshold = AdaptiveThreshold(
                        initial_threshold=AudioConfig.SPEAKER_THRESHOLD
                    )
                else:
                    self.adaptive_threshold = None
                
                print(f"{Fore.GREEN}      ✓ Speaker Verification Ready{Style.RESET_ALL}")
            except FileNotFoundError:
                print(f"{Fore.RED}      ✗ Voice profile not found!{Style.RESET_ALL}")
                print(f"{Fore.YELLOW}      Please run: python3 voice_enrollment.py{Style.RESET_ALL}")
                sys.exit(1)
            except Exception as e:
                print(f"{Fore.RED}      ✗ Speaker Verification Failed: {e}{Style.RESET_ALL}")
                sys.exit(1)
        else:
            self.speaker_verifier = None
            print(f"{Fore.YELLOW}[3/6] Speaker Verification Disabled{Style.RESET_ALL}")
        
        # 4. Initialize Audio Capture
        print(f"{Fore.CYAN}[4/6] Initializing Audio Capture...{Style.RESET_ALL}")
        self.audio_capture = DualAudioCapture(
            sample_rate=AudioConfig.SAMPLE_RATE,
            block_size=AudioConfig.BLOCK_SIZE
        )
        print(f"{Fore.GREEN}      ✓ Audio Capture Ready{Style.RESET_ALL}")
        
        # 5. Load Whisper
        print(f"{Fore.CYAN}[5/6] Loading Whisper {AudioConfig.MODEL_SIZE}...{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}         (First run may download ~3GB){Style.RESET_ALL}")
        try:
            # Try GPU first
            try:
                self.model = WhisperModel(
                    AudioConfig.MODEL_SIZE,
                    device="cuda",
                    compute_type="float16"
                )
                print(f"{Fore.GREEN}      ✓ Whisper Loaded (GPU){Style.RESET_ALL}")
            except:
                # Fallback to CPU
                self.model = WhisperModel(
                    AudioConfig.MODEL_SIZE,
                    device="cpu",
                    compute_type=AudioConfig.COMPUTE_TYPE
                )
                print(f"{Fore.GREEN}      ✓ Whisper Loaded (CPU){Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}      ✗ Whisper Failed: {e}{Style.RESET_ALL}")
            sys.exit(1)
        
        # 6. System Check
        print(f"{Fore.CYAN}[6/6] System Check...{Style.RESET_ALL}")
        self._print_configuration()
        print(f"{Fore.GREEN}      ✓ All Systems Operational{Style.RESET_ALL}")
        
        print(f"\n{Fore.GREEN}{'='*70}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}           TITAN EAR V2 READY{Style.RESET_ALL}")
        print(f"{Fore.GREEN}{'='*70}{Style.RESET_ALL}\n")
    
    def _print_configuration(self):
        """Print current configuration."""
        print(f"\n      {Fore.CYAN}Configuration:{Style.RESET_ALL}")
        print(f"        - Language: {AudioConfig.LANGUAGE}")
        print(f"        - AEC: {'Enabled' if AudioConfig.AEC_ENABLED else 'Disabled'}")
        if AudioConfig.AEC_ENABLED:
            print(f"        - AEC Aggressiveness: {AudioConfig.AEC_AGGRESSIVENESS}")
        print(f"        - Speaker Verification: {'Enabled' if AudioConfig.SPEAKER_VERIFICATION_ENABLED else 'Disabled'}")
        print(f"        - VAD Threshold: {AudioConfig.VAD_THRESHOLD}")
        
        # Check if system audio capture is working
        has_system_audio = self.audio_capture.has_system_audio()
        if AudioConfig.AEC_ENABLED:
            status = "Yes" if has_system_audio else "No"
            color = Fore.GREEN if has_system_audio else Fore.YELLOW
            print(f"        - System Audio Capture: {color}{status}{Style.RESET_ALL}")
            if not has_system_audio:
                print(f"          {Fore.YELLOW}(AEC will be limited without system audio){Style.RESET_ALL}")
    
    def start(self):
        """Start all processing threads."""
        self.running = True
        
        # Start audio capture
        self.audio_capture.start()
        
        # Start processing threads
        self.capture_thread = threading.Thread(target=self._capture_worker, daemon=True)
        self.vad_thread = threading.Thread(target=self._vad_worker, daemon=True)
        
        if self.speaker_verifier:
            self.verification_thread = threading.Thread(target=self._verification_worker, daemon=True)
            self.verification_thread.start()
        
        self.transcribe_thread = threading.Thread(target=self._transcribe_worker, daemon=True)
        
        self.capture_thread.start()
        self.vad_thread.start()
        self.transcribe_thread.start()
        
        print(f"{Fore.GREEN}>>> LISTENING FOR YOUR VOICE...{Style.RESET_ALL}\n")
    
    def stop(self):
        """Stop all processing."""
        self.running = False
        
        # Stop audio capture
        self.audio_capture.stop()
        
        # Wait for threads
        time.sleep(0.5)
        
        # Print statistics
        if self.aec and AudioConfig.AEC_ENABLED:
            print(f"\n{Fore.CYAN}[AEC STATS]{Style.RESET_ALL}")
            stats = self.aec.get_stats()
            for key, value in stats.items():
                print(f"  {key}: {value}")
        
        if self.speaker_verifier and AudioConfig.SPEAKER_VERIFICATION_ENABLED:
            self.speaker_verifier.print_stats()
        
        print(f"{Fore.YELLOW}[SYSTEM] Titan Ear V2 Stopped.{Style.RESET_ALL}")
    
    def _capture_worker(self):
        """
        Thread 1: Capture audio and apply AEC.
        """
        while self.running:
            # Get audio pair (mic, system)
            mic_audio, system_audio = self.audio_capture.get_audio_pair()
            
            if mic_audio is None:
                continue
            
            # Apply AEC if enabled and system audio exists
            if self.aec and system_audio is not None:
                cleaned_audio = self.aec.process(mic_audio, system_audio)
                cleaned_audio = self.noise_gate.process(cleaned_audio)
            else:
                cleaned_audio = mic_audio
            
            # Send to next stage
            self.raw_audio_queue.put(cleaned_audio.flatten().astype(np.float32))
    
    def _vad_worker(self):
        """
        Thread 2: Voice Activity Detection with ring buffer.
        """
        maxlen = int((AudioConfig.SAMPLE_RATE / AudioConfig.BLOCK_SIZE) * 
                     AudioConfig.PRE_ROLL_DURATION)
        ring_buffer = collections.deque(maxlen=maxlen)
        
        triggered = False
        voiced_frames = []
        silence_counter = 0
        
        chunks_per_second = AudioConfig.SAMPLE_RATE / AudioConfig.BLOCK_SIZE
        silence_threshold_chunks = AudioConfig.SILENCE_TIMEOUT * chunks_per_second
        
        while self.running:
            try:
                chunk = self.raw_audio_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            
            # VAD inference
            try:
                chunk_tensor = torch.from_numpy(chunk)
                speech_prob = self.vad_model(chunk_tensor, AudioConfig.SAMPLE_RATE).item()
            except:
                speech_prob = 0.0
            
            current_is_speech = speech_prob > AudioConfig.VAD_THRESHOLD
            
            if not triggered:
                ring_buffer.append(chunk)
                if current_is_speech:
                    print(f"{Fore.YELLOW}🎤 [DETECTING]...{Style.RESET_ALL}", end="\r", flush=True)
                    triggered = True
                    voiced_frames.extend(ring_buffer)
                    voiced_frames.append(chunk)
                    ring_buffer.clear()
                    silence_counter = 0
            else:
                voiced_frames.append(chunk)
                
                if current_is_speech:
                    silence_counter = 0
                else:
                    silence_counter += 1
                
                if silence_counter > silence_threshold_chunks:
                    triggered = False
                    
                    total_duration = (len(voiced_frames) * AudioConfig.BLOCK_SIZE) / AudioConfig.SAMPLE_RATE
                    
                    if total_duration > AudioConfig.MIN_SPEECH_DURATION:
                        full_audio = np.concatenate(voiced_frames)
                        
                        # Send to verification or directly to transcription
                        if self.speaker_verifier:
                            self.verified_audio_queue.put(full_audio)
                        else:
                            self.transcribe_queue.put(full_audio)
                        
                        print(f"{Fore.MAGENTA}⚡ [VERIFYING]...{Style.RESET_ALL}", end="\r", flush=True)
                    
                    voiced_frames = []
                    silence_counter = 0
    
    def _verification_worker(self):
        """
        Thread 3: Speaker Verification.
        """
        while self.running:
            try:
                audio_data = self.verified_audio_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            
            # Verify speaker
            is_user, confidence, similarity = self.speaker_verifier.verify(
                audio_data, AudioConfig.SAMPLE_RATE
            )
            
            # Update adaptive threshold
            if self.adaptive_threshold:
                self.adaptive_threshold.update(is_user)
                new_threshold = self.adaptive_threshold.get_threshold()
                if abs(new_threshold - self.speaker_verifier.threshold) > 0.05:
                    self.speaker_verifier.adjust_threshold(new_threshold)
            
            if is_user:
                # Send to transcription
                self.transcribe_queue.put(audio_data)
                print(f"{Fore.CYAN}✓ [USER VERIFIED - {similarity:.2f}]{Style.RESET_ALL}", end="\r", flush=True)
            else:
                # Reject
                print(f"{Fore.RED}✗ [NOT USER - {similarity:.2f}]        {Style.RESET_ALL}", end="\r", flush=True)
    
    def _transcribe_worker(self):
        """
        Thread 4: Whisper Transcription.
        """
        while self.running:
            try:
                audio_data = self.transcribe_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            
            try:
                # Whisper inference
                segments, info = self.model.transcribe(
                    audio_data,
                    beam_size=AudioConfig.BEAM_SIZE,
                    language=AudioConfig.LANGUAGE,
                    condition_on_previous_text=False,
                    initial_prompt=AudioConfig.INITIAL_PROMPT,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=500)
                )
                
                text = " ".join([segment.text for segment in segments]).strip()
                
                # Filter hallucinations
                bad_phrases = ["Thank you.", "Thanks for watching", "You", ".", "Merci."]
                
                if text and text not in bad_phrases and len(text) > 1:
                    print(f"\n{Fore.GREEN}📝 [TRANSCRIPTION]: {Style.BRIGHT}{text}{Style.RESET_ALL}\n")
                    
                    if self.callback:
                        self.callback(text)
                else:
                    print(f"{Fore.BLACK}⌀ [EMPTY]               {Style.RESET_ALL}", end="\r", flush=True)
            
            except Exception as e:
                print(f"\n{Fore.RED}[TRANSCRIPTION ERROR] {e}{Style.RESET_ALL}")

# ==================================================================================
# EXAMPLE USAGE
# ==================================================================================

def main_agent_loop(text):
    """
    Hook for your agent logic.
    """
    cmd = text.lower()
    
    if "arrête" in cmd or "stop" in cmd or "quitter" in cmd:
        print(f"{Fore.RED}>>> COMMANDE D'ARRÊT REÇUE{Style.RESET_ALL}")
        return False
    
    # Add your custom commands here
    # if "navigateur" in cmd:
    #     os.system("firefox")
    
    return True


if __name__ == "__main__":
    # Check for voice profile
    if AudioConfig.SPEAKER_VERIFICATION_ENABLED and not os.path.exists("user_voice_profile.pkl"):
        print(f"{Fore.RED}ERROR: Voice profile not found!{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Please run: python3 voice_enrollment.py{Style.RESET_ALL}")
        sys.exit(1)
    
    # Callback
    running = True
    def on_hear(text):
        global running
        should_continue = main_agent_loop(text)
        if not should_continue:
            running = False
    
    # Start Titan Ear V2
    ear = TitanEarV2(on_speech_detected_callback=on_hear)
    ear.start()
    
    # Main loop
    try:
        while running:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}[INTERRUPT] Shutting down...{Style.RESET_ALL}")
    finally:
        ear.stop()