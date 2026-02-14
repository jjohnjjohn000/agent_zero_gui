#!/usr/bin/env python3
"""
TITAN EAR V2 - Enhanced Voice Enrollment
=========================================
Records multiple samples with variety for robust speaker verification.
"""

import os
import sys
import time
import numpy as np
import sounddevice as sd
import torch
from pyannote.audio import Model
from pyannote.audio import Inference
from colorama import init, Fore, Style
import pickle

init(autoreset=True)

# Configuration
SAMPLE_RATE = 16000
NUM_SAMPLES = 10              # Number of different recordings
SAMPLE_DURATION = 30         # Seconds per sample (shorter but more varied)
OUTPUT_FILE = "user_voice_profile.pkl"

class EnhancedVoiceEnroller:
    def __init__(self):
        print(f"{Fore.CYAN}[SYSTEM] Enhanced Voice Enrollment System{Style.RESET_ALL}")
        print(f"{Fore.CYAN}[SYSTEM] Will create profile from {NUM_SAMPLES} varied samples{Style.RESET_ALL}\n")
        
        # Get HuggingFace token
        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        
        # Load speaker embedding model
        print(f"{Fore.CYAN}[SYSTEM] Loading Pyannote embedding model...{Style.RESET_ALL}")
        try:
            self.model = Model.from_pretrained(
                "pyannote/embedding",
                use_auth_token=hf_token
            )
            self.inference = Inference(self.model, window="whole")
            print(f"{Fore.GREEN}[SYSTEM] ✓ Embedding model loaded{Style.RESET_ALL}\n")
        except Exception as e:
            print(f"{Fore.RED}[ERROR] Failed to load model: {e}{Style.RESET_ALL}")
            print(f"\n{Fore.YELLOW}Fix: Set HuggingFace token{Style.RESET_ALL}")
            print(f"  export HF_TOKEN='your_token_here'")
            sys.exit(1)
    
    def record_sample(self, sample_num, duration, instructions):
        """Record one voice sample with specific instructions."""
        print(f"\n{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}SAMPLE {sample_num}/{NUM_SAMPLES} - {duration} seconds{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}\n")
        
        print(f"{Fore.YELLOW}{instructions}{Style.RESET_ALL}\n")
        
        input(f"{Fore.GREEN}Press ENTER when ready to record sample {sample_num}...{Style.RESET_ALL}")
        
        print(f"\n{Fore.RED}🔴 RECORDING SAMPLE {sample_num}...{Style.RESET_ALL}")
        
        # Record with progress
        audio_data = []
        
        def callback(indata, frames, time_info, status):
            if status:
                print(f"{Fore.RED}[WARNING] {status}{Style.RESET_ALL}")
            audio_data.append(indata.copy())
        
        with sd.InputStream(samplerate=SAMPLE_RATE, 
                           channels=1, 
                           callback=callback):
            for remaining in range(duration, 0, -1):
                print(f"\r{Fore.YELLOW}⏱️  Sample {sample_num}: {remaining} seconds remaining...{Style.RESET_ALL}", 
                      end='', flush=True)
                time.sleep(1)
        
        print(f"\n{Fore.GREEN}✓ Sample {sample_num} complete!{Style.RESET_ALL}")
        
        # Concatenate
        audio = np.concatenate(audio_data, axis=0).flatten()
        return audio
    
    def create_embedding(self, audio):
        """Create speaker embedding from audio."""
        audio_tensor = torch.from_numpy(audio).float()
        
        with torch.no_grad():
            embedding = self.inference({
                "waveform": audio_tensor.unsqueeze(0), 
                "sample_rate": SAMPLE_RATE
            })
        
        return embedding
    
    def run_enhanced_enrollment(self):
        """Run multi-sample enrollment."""
        print(f"\n{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}  ENHANCED VOICE ENROLLMENT - Multiple Sample Strategy{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}\n")
        
        print(f"{Fore.GREEN}Strategy:{Style.RESET_ALL}")
        print(f"  • {NUM_SAMPLES} different recordings")
        print(f"  • {SAMPLE_DURATION} seconds each")
        print(f"  • Varied speaking styles")
        print(f"  • Creates robust voice profile\n")
        
        print(f"{Fore.YELLOW}Why this works better:{Style.RESET_ALL}")
        print(f"  • Captures your voice in different states")
        print(f"  • Different volumes, pitches, emotions")
        print(f"  • More representative of real-world usage")
        print(f"  • Reduces false rejections\n")
        
        input(f"{Fore.GREEN}Press ENTER to begin enrollment...{Style.RESET_ALL}")
        
        # Define different speaking scenarios
        scenarios = [
            {
                "duration": 30,
                "instructions": (
                    "📖 SAMPLE 1: Read naturally\n"
                    "   Read aloud from a book, article, or describe your day.\n"
                    "   Speak at NORMAL volume and pace."
                )
            },
            {
                "duration": 30,
                "instructions": (
                    "🔢 SAMPLE 2: Numbers and commands\n"
                    "   Count from 1-50, say dates, give system commands.\n"
                    "   Example: 'Execute script. Install package. Initialize system.'\n"
                    "   Speak CLEARLY and deliberately."
                )
            },
            {
                "duration": 30,
                "instructions": (
                    "🎭 SAMPLE 3: Varied pitch and volume\n"
                    "   Speak QUIETLY for 5 seconds, then NORMAL, then LOUDER.\n"
                    "   Vary your pitch - high, normal, low.\n"
                    "   Talk about anything - weather, plans, thoughts."
                )
            },
            {
                "duration": 30,
                "instructions": (
                    "💬 SAMPLE 4: Conversational\n"
                    "   Speak as if talking to a friend.\n"
                    "   Use natural pauses, 'umm', 'euh', breathing.\n"
                    "   Tell a story or explain something you're interested in."
                )
            },
            {
                "duration": 30,
                "instructions": (
                    "🗣️ SAMPLE 5: Technical speech\n"
                    "   Give technical commands and speak terminology.\n"
                    "   Example: 'Python script. Neural network. Database query.'\n"
                    "   Mix with normal conversation."
                )
            }
        ]
        
        # Record all samples
        all_audio = []
        all_embeddings = []
        
        for i, scenario in enumerate(scenarios[:NUM_SAMPLES], 1):
            audio = self.record_sample(i, scenario["duration"], scenario["instructions"])
            all_audio.append(audio)
            
            # Create embedding
            print(f"{Fore.CYAN}[PROCESSING] Creating embedding for sample {i}...{Style.RESET_ALL}")
            embedding = self.create_embedding(audio)
            all_embeddings.append(embedding)
            print(f"{Fore.GREEN}[PROCESSING] ✓ Embedding {i} created{Style.RESET_ALL}")
        
        # Average embeddings for robustness
        print(f"\n{Fore.CYAN}[PROCESSING] Combining embeddings...{Style.RESET_ALL}")
        
        # Convert to numpy if needed
        embeddings_np = []
        for emb in all_embeddings:
            if torch.is_tensor(emb):
                embeddings_np.append(emb.cpu().numpy())
            else:
                embeddings_np.append(emb)
        
        # Stack and average
        stacked = np.stack([e.flatten() for e in embeddings_np])
        averaged_embedding = np.mean(stacked, axis=0)
        
        print(f"{Fore.GREEN}[PROCESSING] ✓ Created robust averaged embedding{Style.RESET_ALL}")
        
        # Save profile
        profile = {
            "embedding": averaged_embedding,
            "individual_embeddings": embeddings_np,  # Keep individual ones too
            "sample_rate": SAMPLE_RATE,
            "model": "pyannote/embedding",
            "num_samples": NUM_SAMPLES,
            "created_at": time.time()
        }
        
        with open(OUTPUT_FILE, 'wb') as f:
            pickle.dump(profile, f)
        
        print(f"\n{Fore.GREEN}[SAVED] Voice profile saved to: {OUTPUT_FILE}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}[INFO] File size: {os.path.getsize(OUTPUT_FILE)} bytes{Style.RESET_ALL}")
        print(f"{Fore.CYAN}[INFO] Contains {NUM_SAMPLES} sample embeddings{Style.RESET_ALL}")
        
        # Quality check
        print(f"\n{Fore.CYAN}[QUALITY CHECK] Verifying profile quality...{Style.RESET_ALL}")
        self._quality_check(embeddings_np, averaged_embedding)
        
        print(f"\n{Fore.GREEN}{'='*70}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}✓ ENHANCED ENROLLMENT COMPLETE!{Style.RESET_ALL}")
        print(f"{Fore.GREEN}{'='*70}{Style.RESET_ALL}\n")
        print(f"{Fore.CYAN}Your robust voice profile is ready!{Style.RESET_ALL}")
        print(f"{Fore.CYAN}This should significantly reduce false rejections.{Style.RESET_ALL}\n")
    
    def _quality_check(self, embeddings, averaged):
        """Check quality of enrollment."""
        from scipy.spatial.distance import cosine
        
        # Check similarity between samples
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                emb1 = embeddings[i].flatten()
                emb2 = embeddings[j].flatten()
                sim = 1 - cosine(emb1, emb2)
                similarities.append(sim)
        
        avg_similarity = np.mean(similarities)
        min_similarity = np.min(similarities)
        
        print(f"  Average inter-sample similarity: {avg_similarity:.3f}")
        print(f"  Minimum inter-sample similarity: {min_similarity:.3f}")
        
        if avg_similarity > 0.7:
            print(f"{Fore.GREEN}  ✓ Excellent consistency!{Style.RESET_ALL}")
        elif avg_similarity > 0.6:
            print(f"{Fore.YELLOW}  ⚠ Good, but could re-enroll for better results{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}  ✗ Low consistency - consider re-enrolling{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}  Tip: Speak more consistently across samples{Style.RESET_ALL}")


def add_to_existing_profile():
    """Add more samples to existing profile."""
    if not os.path.exists(OUTPUT_FILE):
        print(f"{Fore.RED}[ERROR] No existing profile found!{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Run normal enrollment first.{Style.RESET_ALL}")
        return
    
    print(f"{Fore.CYAN}[SYSTEM] Loading existing profile...{Style.RESET_ALL}")
    
    with open(OUTPUT_FILE, 'rb') as f:
        profile = pickle.load(f)
    
    existing_embeddings = profile.get('individual_embeddings', [profile['embedding']])
    
    print(f"{Fore.GREEN}[SYSTEM] ✓ Loaded profile with {len(existing_embeddings)} samples{Style.RESET_ALL}\n")
    print(f"{Fore.YELLOW}You can add more samples to improve robustness.{Style.RESET_ALL}\n")
    
    num_new = int(input(f"{Fore.CYAN}How many additional samples? (1-5): {Style.RESET_ALL}"))
    num_new = max(1, min(5, num_new))
    
    enroller = EnhancedVoiceEnroller()
    
    new_embeddings = []
    for i in range(num_new):
        audio = enroller.record_sample(
            i + 1, 
            15, 
            f"Additional sample {i+1}: Speak naturally, vary your voice."
        )
        
        embedding = enroller.create_embedding(audio)
        if torch.is_tensor(embedding):
            embedding = embedding.cpu().numpy()
        new_embeddings.append(embedding)
    
    # Combine with existing
    all_embeddings = existing_embeddings + new_embeddings
    stacked = np.stack([e.flatten() for e in all_embeddings])
    new_averaged = np.mean(stacked, axis=0)
    
    # Save updated profile
    profile['embedding'] = new_averaged
    profile['individual_embeddings'] = all_embeddings
    profile['num_samples'] = len(all_embeddings)
    profile['updated_at'] = time.time()
    
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(profile, f)
    
    print(f"\n{Fore.GREEN}✓ Profile updated with {len(all_embeddings)} total samples!{Style.RESET_ALL}\n")


def main():
    print(f"\n{Fore.CYAN}TITAN EAR V2 - Enhanced Voice Enrollment{Style.RESET_ALL}\n")
    
    if os.path.exists(OUTPUT_FILE):
        print(f"{Fore.YELLOW}Existing voice profile found.{Style.RESET_ALL}\n")
        print(f"{Fore.CYAN}Options:{Style.RESET_ALL}")
        print(f"  1. Create NEW profile (recommended for best results)")
        print(f"  2. Add samples to EXISTING profile")
        print(f"  3. Cancel\n")
        
        choice = input(f"{Fore.GREEN}Choice (1/2/3): {Style.RESET_ALL}")
        
        if choice == "1":
            response = input(f"{Fore.YELLOW}This will overwrite existing profile. Continue? (yes/no): {Style.RESET_ALL}")
            if response.lower() not in ['yes', 'y']:
                print(f"{Fore.CYAN}[CANCELLED]{Style.RESET_ALL}")
                return
        elif choice == "2":
            add_to_existing_profile()
            return
        else:
            print(f"{Fore.CYAN}[CANCELLED]{Style.RESET_ALL}")
            return
    
    # Run enhanced enrollment
    enroller = EnhancedVoiceEnroller()
    enroller.run_enhanced_enrollment()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}[CANCELLED] Enrollment interrupted.{Style.RESET_ALL}")
        sys.exit(0)
    except Exception as e:
        print(f"\n{Fore.RED}[ERROR] {e}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()
        sys.exit(1)