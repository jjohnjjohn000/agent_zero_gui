#!/usr/bin/env python3
"""
TITAN EAR V2 - WORKING Acoustic Echo Cancellation
==================================================
Proper AEC with delay estimation and robust adaptive filtering.
"""

import numpy as np
from scipy import signal
from scipy.signal import correlate
from collections import deque
from colorama import Fore, Style

class WorkingAECProcessor:
    """
    Properly working AEC with:
    1. Automatic delay detection
    2. Robust adaptive filtering
    3. Voice preservation
    """
    
    def __init__(self, sample_rate=16000, filter_length=2048, mu=0.5):
        """
        :param sample_rate: Audio sample rate
        :param filter_length: Adaptive filter length (longer = better echo removal)
        :param mu: Step size (0.1 - 1.0, higher = faster adaptation)
        """
        self.sample_rate = sample_rate
        self.filter_length = filter_length
        self.mu = mu
        
        # Adaptive filter (Wiener filter approach)
        self.W = np.zeros(filter_length, dtype=np.float64)
        
        # Reference buffer (system audio history)
        self.ref_buffer = deque(maxlen=filter_length * 2)
        
        # Delay estimation
        self.estimated_delay = 0
        self.delay_estimated = False
        
        # Power estimates for normalization
        self.ref_power = 1e-10
        self.mic_power = 1e-10
        
        # Statistics
        self.frames_processed = 0
        self.total_reduction_db = 0
        
        print(f"{Fore.CYAN}[AEC] Working AEC initialized (filter_len={filter_length}, mu={mu}){Style.RESET_ALL}")
    
    def estimate_delay(self, mic_signal, ref_signal):
        """
        Estimate the delay between reference and microphone using cross-correlation.
        """
        if len(mic_signal) < 1000 or len(ref_signal) < 1000:
            return 0
        
        # Use only a portion for speed
        mic_sample = mic_signal[:min(4000, len(mic_signal))]
        ref_sample = ref_signal[:min(4000, len(ref_signal))]
        
        # Cross-correlation
        correlation = correlate(mic_sample, ref_sample, mode='full')
        
        # Find peak
        delay = np.argmax(np.abs(correlation)) - len(ref_sample) + 1
        
        # Limit delay to reasonable range (0-500ms)
        max_delay = int(0.5 * self.sample_rate)  # 500ms
        delay = np.clip(delay, 0, max_delay)
        
        return delay
    
    def process(self, mic_signal, reference_signal):
        """
        Process audio with proper AEC.
        
        :param mic_signal: Microphone input (voice + echo)
        :param reference_signal: System audio (the echo source)
        :return: Cleaned signal
        """
        if reference_signal is None or len(reference_signal) == 0:
            return mic_signal
        
        # Ensure 1D
        mic_signal = mic_signal.flatten().astype(np.float64)
        reference_signal = reference_signal.flatten().astype(np.float64)
        
        # Make same length
        max_len = max(len(mic_signal), len(reference_signal))
        if len(mic_signal) < max_len:
            mic_signal = np.pad(mic_signal, (0, max_len - len(mic_signal)))
        if len(reference_signal) < max_len:
            reference_signal = np.pad(reference_signal, (0, max_len - len(reference_signal)))
        
        # Estimate delay on first few frames
        if not self.delay_estimated and len(self.ref_buffer) > 1000:
            accumulated_ref = np.array(list(self.ref_buffer))
            self.estimated_delay = self.estimate_delay(mic_signal, accumulated_ref)
            self.delay_estimated = True
            print(f"{Fore.CYAN}[AEC] Estimated delay: {self.estimated_delay} samples ({self.estimated_delay/self.sample_rate*1000:.1f}ms){Style.RESET_ALL}")
        
        # Apply estimated delay to reference
        if self.estimated_delay > 0 and self.estimated_delay < len(reference_signal):
            reference_signal = np.roll(reference_signal, -self.estimated_delay)
        
        # Update power estimates (for normalization)
        self.ref_power = 0.95 * self.ref_power + 0.05 * np.mean(reference_signal**2)
        self.mic_power = 0.95 * self.mic_power + 0.05 * np.mean(mic_signal**2)
        
        # Process sample by sample
        output = np.zeros(len(mic_signal), dtype=np.float64)
        
        for i in range(len(mic_signal)):
            # Add to buffer
            self.ref_buffer.append(reference_signal[i])
            
            # Need enough history
            if len(self.ref_buffer) < self.filter_length:
                output[i] = mic_signal[i]
                continue
            
            # Get reference vector (most recent samples)
            X = np.array(list(self.ref_buffer)[-self.filter_length:], dtype=np.float64)
            
            # Estimate echo
            y_hat = np.dot(self.W, X)
            
            # Error signal (desired output)
            e = mic_signal[i] - y_hat
            output[i] = e
            
            # Update filter (NLMS with regularization)
            X_power = np.dot(X, X) + 1e-4  # Regularization
            step = (self.mu * e) / X_power
            self.W += step * X
            
            # Prevent filter divergence
            filter_energy = np.dot(self.W, self.W)
            if filter_energy > 100:  # Limit filter energy
                self.W *= 0.9
        
        self.frames_processed += 1
        
        # Stats
        if self.frames_processed % 100 == 0:
            self._calculate_stats(mic_signal, output)
        
        return output
    
    def _calculate_stats(self, original, cleaned):
        """Calculate reduction statistics."""
        try:
            orig_power = np.mean(original**2) + 1e-10
            clean_power = np.mean(cleaned**2) + 1e-10
            
            reduction_db = 10 * np.log10(orig_power / clean_power)
            self.total_reduction_db = reduction_db
            
            if self.frames_processed % 500 == 0:
                print(f"{Fore.CYAN}[AEC] Reduction: {reduction_db:.1f} dB, Filter norm: {np.linalg.norm(self.W):.2f}{Style.RESET_ALL}")
        except:
            pass
    
    def reset(self):
        """Reset adaptive state."""
        self.W *= 0.95
    
    def get_stats(self):
        """Get statistics."""
        return {
            "frames_processed": self.frames_processed,
            "echo_reduction_db": self.total_reduction_db,
            "filter_norm": np.linalg.norm(self.W),
            "estimated_delay_ms": (self.estimated_delay / self.sample_rate) * 1000,
            "delay_estimated": self.delay_estimated
        }


class RobustSpectralAEC:
    """
    Spectral subtraction with voice preservation.
    More aggressive than adaptive filtering.
    """
    
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        
        # Spectral subtraction parameters
        self.over_subtraction = 3.0  # How much to subtract
        self.spectral_floor = 0.1    # Minimum gain (preserves voice)
        
        # For smoothing
        self.prev_gain = None
        self.smoothing = 0.7  # Temporal smoothing
        
        self.frames_processed = 0
        
        print(f"{Fore.CYAN}[AEC] Robust Spectral AEC initialized{Style.RESET_ALL}")
    
    def process(self, mic_signal, reference_signal):
        """
        Spectral subtraction with voice preservation.
        """
        if reference_signal is None or len(reference_signal) == 0:
            return mic_signal
        
        mic_signal = mic_signal.flatten()
        reference_signal = reference_signal.flatten()
        
        # Match lengths
        max_len = max(len(mic_signal), len(reference_signal))
        if len(mic_signal) < max_len:
            mic_signal = np.pad(mic_signal, (0, max_len - len(mic_signal)))
        if len(reference_signal) < max_len:
            reference_signal = np.pad(reference_signal, (0, max_len - len(reference_signal)))
        
        # FFT
        Mic = np.fft.rfft(mic_signal)
        Ref = np.fft.rfft(reference_signal)
        
        # Magnitudes
        Mic_mag = np.abs(Mic)
        Ref_mag = np.abs(Ref)
        
        # Phase (preserve from mic)
        Mic_phase = np.angle(Mic)
        
        # Spectral subtraction
        # Gain = max(1 - alpha * |Ref| / |Mic|, floor)
        gain = np.ones_like(Mic_mag)
        
        # Only subtract where reference is significant
        mask = Mic_mag > 1e-6
        gain[mask] = 1.0 - (self.over_subtraction * Ref_mag[mask] / Mic_mag[mask])
        gain = np.maximum(gain, self.spectral_floor)
        
        # Smooth over time
        if self.prev_gain is not None and len(self.prev_gain) == len(gain):
            gain = self.smoothing * self.prev_gain + (1 - self.smoothing) * gain
        self.prev_gain = gain.copy()
        
        # Apply gain
        Clean_mag = Mic_mag * gain
        
        # Reconstruct
        Clean = Clean_mag * np.exp(1j * Mic_phase)
        output = np.fft.irfft(Clean, n=len(mic_signal))
        
        self.frames_processed += 1
        
        return output
    
    def reset(self):
        self.prev_gain = None
    
    def get_stats(self):
        return {
            "frames_processed": self.frames_processed,
            "method": "spectral_subtraction"
        }


# Alias
AECProcessor = WorkingAECProcessor


def test_with_real_delay():
    """Test with realistic acoustic delay."""
    from colorama import init
    init(autoreset=True)
    
    print(f"{Fore.CYAN}Testing AEC with realistic acoustic echo...{Style.RESET_ALL}\n")
    
    sr = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration))
    
    # System audio (music/video)
    freq_music = 440
    system_audio = np.sin(2 * np.pi * freq_music * t) * 0.5
    
    # Add harmonics (more realistic)
    system_audio += 0.3 * np.sin(2 * np.pi * freq_music * 2 * t)
    system_audio += 0.2 * np.sin(2 * np.pi * freq_music * 3 * t)
    
    # Your voice (different frequency)
    freq_voice = 200
    voice = np.sin(2 * np.pi * freq_voice * t) * 0.4
    voice += 0.2 * np.sin(2 * np.pi * freq_voice * 1.5 * t)
    
    # Realistic echo: delayed + attenuated system audio
    delay_samples = int(0.05 * sr)  # 50ms delay (realistic for room acoustics)
    echo = np.roll(system_audio, delay_samples) * 0.6  # 60% echo strength
    
    # Microphone = voice + echo
    mic_with_echo = voice + echo
    
    print(f"Test setup:")
    print(f"  Voice frequency: {freq_voice} Hz")
    print(f"  System audio frequency: {freq_music} Hz")
    print(f"  Echo delay: {delay_samples/sr*1000:.1f} ms")
    print(f"  Echo strength: 60%\n")
    
    # Test both AEC methods
    print(f"{Fore.YELLOW}Testing Working AEC...{Style.RESET_ALL}")
    aec1 = WorkingAECProcessor(filter_length=2048, mu=0.8)
    
    chunk_size = 512
    cleaned1_chunks = []
    
    for i in range(0, len(mic_with_echo) - chunk_size, chunk_size):
        mic_chunk = mic_with_echo[i:i+chunk_size]
        sys_chunk = system_audio[i:i+chunk_size]
        
        cleaned = aec1.process(mic_chunk, sys_chunk)
        cleaned1_chunks.append(cleaned)
    
    cleaned1 = np.concatenate(cleaned1_chunks)
    
    # Calculate performance
    voice_len = len(cleaned1)
    voice_power = np.mean(voice[:voice_len]**2)
    echo_power = np.mean(echo[:voice_len]**2)
    residual_power = np.mean((cleaned1 - voice[:voice_len])**2)
    
    original_snr = 10 * np.log10(voice_power / echo_power)
    cleaned_snr = 10 * np.log10(voice_power / (residual_power + 1e-10))
    improvement = cleaned_snr - original_snr
    
    print(f"\n{Fore.GREEN}Results:{Style.RESET_ALL}")
    print(f"  Original SNR: {original_snr:.1f} dB")
    print(f"  Cleaned SNR: {cleaned_snr:.1f} dB")
    print(f"  Improvement: {improvement:.1f} dB")
    
    if improvement > 5:
        print(f"{Fore.GREEN}  ✓ AEC WORKING WELL!{Style.RESET_ALL}")
    elif improvement > 0:
        print(f"{Fore.YELLOW}  ~ AEC working but could be better{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}  ✗ AEC not helping{Style.RESET_ALL}")
    
    stats = aec1.get_stats()
    print(f"\n{Fore.CYAN}AEC Stats:{Style.RESET_ALL}")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    
    # Test spectral method
    print(f"\n{Fore.YELLOW}Testing Spectral AEC...{Style.RESET_ALL}")
    aec2 = RobustSpectralAEC()
    
    cleaned2_chunks = []
    for i in range(0, len(mic_with_echo) - chunk_size, chunk_size):
        mic_chunk = mic_with_echo[i:i+chunk_size]
        sys_chunk = system_audio[i:i+chunk_size]
        
        cleaned = aec2.process(mic_chunk, sys_chunk)
        cleaned2_chunks.append(cleaned)
    
    cleaned2 = np.concatenate(cleaned2_chunks)
    
    residual_power2 = np.mean((cleaned2[:voice_len] - voice[:voice_len])**2)
    cleaned_snr2 = 10 * np.log10(voice_power / (residual_power2 + 1e-10))
    improvement2 = cleaned_snr2 - original_snr
    
    print(f"\n{Fore.GREEN}Spectral Results:{Style.RESET_ALL}")
    print(f"  Cleaned SNR: {cleaned_snr2:.1f} dB")
    print(f"  Improvement: {improvement2:.1f} dB")
    
    if improvement2 > improvement:
        print(f"{Fore.GREEN}  ✓ Spectral method works better!{Style.RESET_ALL}")
    
    return improvement > 5


if __name__ == "__main__":
    success = test_with_real_delay()
    if not success:
        print(f"\n{Fore.YELLOW}Note: Tune parameters for your specific audio setup{Style.RESET_ALL}")