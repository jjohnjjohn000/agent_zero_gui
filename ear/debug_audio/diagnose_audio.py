#!/usr/bin/env python3
"""
Audio Device Diagnostic
Helps identify why system audio capture isn't working
"""

import sounddevice as sd
from colorama import Fore, Style, init

init(autoreset=True)

print(f"{Fore.CYAN}=== AUDIO DEVICE DIAGNOSTIC ==={Style.RESET_ALL}\n")

devices = sd.query_devices()

print(f"{Fore.GREEN}All Available Devices:{Style.RESET_ALL}")
print(f"{'-'*80}")

for idx, device in enumerate(devices):
    print(f"\n{Fore.YELLOW}[{idx}] {device['name']}{Style.RESET_ALL}")
    print(f"    Max Input Channels: {device['max_input_channels']}")
    print(f"    Max Output Channels: {device['max_output_channels']}")
    print(f"    Default Sample Rate: {device['default_samplerate']}")
    
    # Check if it's a potential monitor
    name_lower = device['name'].lower()
    is_monitor = 'monitor' in name_lower
    is_loopback = 'loopback' in name_lower
    has_input = device['max_input_channels'] > 0
    
    if is_monitor and has_input:
        print(f"    {Fore.GREEN}>>> MONITOR DEVICE (can capture system audio){Style.RESET_ALL}")
    elif is_loopback and has_input:
        print(f"    {Fore.GREEN}>>> LOOPBACK DEVICE (can capture system audio){Style.RESET_ALL}")
    elif has_input and not is_monitor:
        print(f"    >>> Regular input device")

print(f"\n{'-'*80}")
print(f"\n{Fore.CYAN}Looking for Monitor/Loopback devices...{Style.RESET_ALL}")

# Find monitors
monitors = []
for idx, device in enumerate(devices):
    name_lower = device['name'].lower()
    if device['max_input_channels'] > 0:
        if 'monitor' in name_lower or 'loopback' in name_lower:
            monitors.append((idx, device['name']))

if monitors:
    print(f"{Fore.GREEN}✓ Found {len(monitors)} monitor device(s):{Style.RESET_ALL}")
    for idx, name in monitors:
        print(f"  [{idx}] {name}")
else:
    print(f"{Fore.RED}✗ No monitor devices found{Style.RESET_ALL}")
    print(f"\n{Fore.YELLOW}This means:{Style.RESET_ALL}")
    print(f"  - PulseAudio is not exposing monitor sources")
    print(f"  - Or they exist but sounddevice can't see them")
    print(f"\n{Fore.CYAN}Solutions:{Style.RESET_ALL}")
    print(f"  1. Check PulseAudio: pactl list sources | grep -i monitor")
    print(f"  2. Restart PulseAudio: pulseaudio -k && pulseaudio --start")
    print(f"  3. Or just use headphones and disable AEC")

print(f"\n{Fore.CYAN}Default Input Device:{Style.RESET_ALL}")
try:
    default_in = sd.query_devices(kind='input')
    print(f"  {default_in['name']}")
except:
    print(f"  {Fore.RED}Could not determine default input{Style.RESET_ALL}")

print(f"\n{Fore.CYAN}Default Output Device:{Style.RESET_ALL}")
try:
    default_out = sd.query_devices(kind='output')
    print(f"  {default_out['name']}")
except:
    print(f"  {Fore.RED}Could not determine default output{Style.RESET_ALL}")

print(f"\n{'-'*80}\n")
