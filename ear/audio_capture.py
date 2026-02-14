#!/usr/bin/env python3
"""
TITAN EAR V2 - Dual Audio Capture System
==========================================
Captures both microphone input AND system audio (loopback) for AEC processing.
Linux-specific using PulseAudio.
"""

import numpy as np
import sounddevice as sd
import queue
import threading
from colorama import Fore, Style

class DualAudioCapture:
    """
    Captures two synchronized audio streams:
    1. Microphone input (what you say)
    2. System audio output (what plays on speakers)
    """
    
    def __init__(self, sample_rate=16000, block_size=512):
        self.sample_rate = sample_rate
        self.block_size = block_size
        
        # Queues for audio data
        self.mic_queue = queue.Queue()
        self.system_queue = queue.Queue()
        
        self.running = False
        self.mic_stream = None
        self.system_stream = None
        
        # Find system audio monitor device
        self.system_device = self._find_monitor_device()
        
    def _find_monitor_device(self):
        """
        Find PulseAudio monitor device for system audio capture.
        This captures what's playing on your speakers.
        """
        devices = sd.query_devices()
        
        # Look for monitor devices (usually contains 'monitor' in name)
        for idx, device in enumerate(devices):
            device_name = device['name'].lower()
            if 'monitor' in device_name and device['max_input_channels'] > 0:
                print(f"{Fore.GREEN}[AUDIO] Found system audio: {device['name']}{Style.RESET_ALL}")
                return idx
        
        # Fallback: look for loopback
        for idx, device in enumerate(devices):
            device_name = device['name'].lower()
            if 'loopback' in device_name and device['max_input_channels'] > 0:
                print(f"{Fore.GREEN}[AUDIO] Found loopback: {device['name']}{Style.RESET_ALL}")
                return idx
        
        print(f"{Fore.YELLOW}[WARNING] No monitor device found. AEC will be disabled.{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}[INFO] To enable system audio capture on Linux:{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}       pactl load-module module-loopback{Style.RESET_ALL}")
        return None
    
    def _mic_callback(self, indata, frames, time_info, status):
        """Microphone audio callback."""
        if status:
            print(f"{Fore.RED}[MIC ERROR] {status}{Style.RESET_ALL}")
        self.mic_queue.put(indata.copy())
    
    def _system_callback(self, indata, frames, time_info, status):
        """System audio callback."""
        if status:
            print(f"{Fore.RED}[SYSTEM AUDIO ERROR] {status}{Style.RESET_ALL}")
        self.system_queue.put(indata.copy())
    
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
        
        # Start system audio stream (if available)
        if self.system_device is not None:
            print(f"{Fore.CYAN}[AUDIO] Starting system audio capture...{Style.RESET_ALL}")
            try:
                self.system_stream = sd.InputStream(
                    device=self.system_device,
                    samplerate=self.sample_rate,
                    channels=1,
                    callback=self._system_callback,
                    blocksize=self.block_size
                )
                self.system_stream.start()
                print(f"{Fore.GREEN}[AUDIO] ✓ System audio active{Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.YELLOW}[WARNING] System audio failed: {e}{Style.RESET_ALL}")
                self.system_stream = None
        else:
            print(f"{Fore.YELLOW}[AUDIO] System audio capture disabled{Style.RESET_ALL}")
    
    def stop(self):
        """Stop both audio streams."""
        self.running = False
        
        if self.mic_stream:
            self.mic_stream.stop()
            self.mic_stream.close()
        
        if self.system_stream:
            self.system_stream.stop()
            self.system_stream.close()
        
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
        return self.system_stream is not None


# Quick test function
def test_capture():
    """Test the dual audio capture system."""
    from colorama import init
    init(autoreset=True)
    
    print(f"{Fore.CYAN}Testing Dual Audio Capture...{Style.RESET_ALL}")
    
    capture = DualAudioCapture()
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
