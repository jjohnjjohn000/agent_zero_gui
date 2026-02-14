#!/usr/bin/env python3
"""
TITAN EAR V2 - Dual Audio Capture System (PulseAudio Native)
==============================================================
Uses pulsectl for direct PulseAudio access to capture monitors.
"""

import numpy as np
import sounddevice as sd
import queue
import threading
import subprocess
from colorama import Fore, Style

class DualAudioCapturePulse:
    """
    Captures two synchronized audio streams using PulseAudio:
    1. Microphone input (what you say)
    2. System audio output (what plays on speakers) - via parecord
    """
    
    def __init__(self, sample_rate=16000, block_size=512, monitor_source=None):
        self.sample_rate = sample_rate
        self.block_size = block_size
        
        # Queues for audio data
        self.mic_queue = queue.Queue()
        self.system_queue = queue.Queue()
        
        self.running = False
        self.mic_stream = None
        self.system_capture_thread = None
        
        # Find or use specified monitor source
        self.monitor_source = monitor_source or self._find_monitor_source()
        
    def _find_monitor_source(self):
        """Find PulseAudio monitor source using pactl."""
        try:
            result = subprocess.run(
                ['pactl', 'list', 'sources', 'short'],
                capture_output=True,
                text=True,
                check=True
            )
            
            # Parse output for monitor sources
            for line in result.stdout.split('\n'):
                if 'monitor' in line.lower() and line.strip():
                    parts = line.split()
                    if len(parts) >= 2:
                        source_name = parts[1]
                        print(f"{Fore.GREEN}[AUDIO] Found system audio: {source_name}{Style.RESET_ALL}")
                        return source_name
            
            print(f"{Fore.YELLOW}[WARNING] No monitor source found. AEC will be disabled.{Style.RESET_ALL}")
            return None
            
        except Exception as e:
            print(f"{Fore.YELLOW}[WARNING] Could not query PulseAudio: {e}{Style.RESET_ALL}")
            return None
    
    def _mic_callback(self, indata, frames, time_info, status):
        """Microphone audio callback."""
        if status:
            print(f"{Fore.RED}[MIC ERROR] {status}{Style.RESET_ALL}")
        self.mic_queue.put(indata.copy())
    
    def _system_capture_worker(self):
        """Capture system audio using parecord."""
        if not self.monitor_source:
            return
        
        try:
            # Start parecord process
            cmd = [
                'parecord',
                '--device', self.monitor_source,
                '--rate', str(self.sample_rate),
                '--channels', '1',
                '--format', 's16le',
                '--raw'
            ]
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=self.block_size * 2  # 2 bytes per sample (s16le)
            )
            
            bytes_per_block = self.block_size * 2  # 16-bit = 2 bytes per sample
            
            while self.running:
                # Read audio data
                raw_data = process.stdout.read(bytes_per_block)
                if not raw_data:
                    break
                
                # Convert to numpy array
                audio_array = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
                
                # Reshape to match expected format
                audio_array = audio_array.reshape(-1, 1)
                
                self.system_queue.put(audio_array)
            
            process.terminate()
            process.wait()
            
        except Exception as e:
            print(f"{Fore.RED}[SYSTEM AUDIO ERROR] {e}{Style.RESET_ALL}")
    
    def start(self):
        """Start both audio streams."""
        self.running = True
        
        # Start microphone stream
        print(f"{Fore.CYAN}[AUDIO] Starting microphone capture...{Style.RESET_ALL}")
        self.mic_stream = sd.InputStream(
            device=None,  # Default mic
            samplerate=self.sample_rate,
            channels=1,
            callback=self._mic_callback,
            blocksize=self.block_size
        )
        self.mic_stream.start()
        print(f"{Fore.GREEN}[AUDIO] ✓ Microphone active{Style.RESET_ALL}")
        
        # Start system audio capture (if available)
        if self.monitor_source:
            print(f"{Fore.CYAN}[AUDIO] Starting system audio capture...{Style.RESET_ALL}")
            self.system_capture_thread = threading.Thread(
                target=self._system_capture_worker,
                daemon=True
            )
            self.system_capture_thread.start()
            print(f"{Fore.GREEN}[AUDIO] ✓ System audio active{Style.RESET_ALL}")
        else:
            print(f"{Fore.YELLOW}[AUDIO] System audio capture disabled{Style.RESET_ALL}")
    
    def stop(self):
        """Stop both audio streams."""
        self.running = False
        
        if self.mic_stream:
            self.mic_stream.stop()
            self.mic_stream.close()
        
        if self.system_capture_thread:
            self.system_capture_thread.join(timeout=1.0)
        
        print(f"{Fore.YELLOW}[AUDIO] Capture stopped{Style.RESET_ALL}")
    
    def get_audio_pair(self, timeout=1.0):
        """
        Get synchronized audio from both mic and system.
        Returns: (mic_audio, system_audio) or (mic_audio, None) if no system audio
        """
        try:
            mic_audio = self.mic_queue.get(timeout=timeout)
            
            # Try to get system audio (may not exist)
            try:
                system_audio = self.system_queue.get(timeout=0.01)
            except queue.Empty:
                system_audio = None
            
            return mic_audio, system_audio
        
        except queue.Empty:
            return None, None
    
    def has_system_audio(self):
        """Check if system audio capture is available."""
        return self.monitor_source is not None


# Quick test function
def test_capture():
    """Test the dual audio capture system."""
    from colorama import init
    init(autoreset=True)
    
    print(f"{Fore.CYAN}Testing Dual Audio Capture (PulseAudio Native)...{Style.RESET_ALL}")
    
    capture = DualAudioCapturePulse()
    capture.start()
    
    print(f"{Fore.GREEN}Recording for 5 seconds...{Style.RESET_ALL}")
    
    import time
    start_time = time.time()
    mic_chunks = 0
    system_chunks = 0
    
    while time.time() - start_time < 5:
        mic, sys = capture.get_audio_pair()
        if mic is not None:
            mic_chunks += 1
        if sys is not None:
            system_chunks += 1
    
    capture.stop()
    
    print(f"{Fore.GREEN}Test complete:{Style.RESET_ALL}")
    print(f"  Mic chunks: {mic_chunks}")
    print(f"  System chunks: {system_chunks}")
    print(f"  Has system audio: {capture.has_system_audio()}")


if __name__ == "__main__":
    test_capture()
