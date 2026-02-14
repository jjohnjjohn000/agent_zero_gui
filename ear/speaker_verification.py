#!/usr/bin/env python3
"""
TITAN EAR V2 - Speaker Verification System
============================================
Verifies that the speaker is the enrolled user using voice embeddings.
"""

import numpy as np
import torch
import pickle
import os
from pyannote.audio import Model, Inference
from scipy.spatial.distance import cosine
from colorama import Fore, Style

class SpeakerVerifier:
    """
    Verifies if incoming audio matches the enrolled user's voice.
    Uses cosine similarity between embeddings.
    """
    
    def __init__(self, profile_path="user_voice_profile.pkl", threshold=0.6):
        """
        :param profile_path: Path to the enrolled voice profile
        :param threshold: Similarity threshold (0-1, higher = stricter)
                         Typical values: 0.5 (loose), 0.6 (balanced), 0.75 (strict)
        """
        self.threshold = threshold
        self.profile_path = profile_path
        
        # Load enrolled profile
        print(f"{Fore.CYAN}[SPEAKER] Loading voice profile: {profile_path}{Style.RESET_ALL}")
        if not os.path.exists(profile_path):
            raise FileNotFoundError(
                f"Voice profile not found: {profile_path}\n"
                f"Please run voice_enrollment.py first!"
            )
        
        with open(profile_path, 'rb') as f:
            self.profile = pickle.load(f)
        
        self.enrolled_embedding = self.profile['embedding']
        print(f"{Fore.GREEN}[SPEAKER] ✓ Voice profile loaded{Style.RESET_ALL}")
        
        # Load embedding model
        print(f"{Fore.CYAN}[SPEAKER] Loading embedding model...{Style.RESET_ALL}")
        try:
            self.model = Model.from_pretrained("pyannote/embedding", 
                                              use_auth_token=None)
            self.inference = Inference(self.model, window="whole")
            print(f"{Fore.GREEN}[SPEAKER] ✓ Embedding model ready{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}[ERROR] Failed to load embedding model: {e}{Style.RESET_ALL}")
            raise
        
        # Statistics
        self.verifications_performed = 0
        self.accepted = 0
        self.rejected = 0
        self.similarity_history = []
    
    def verify(self, audio, sample_rate=16000):
        """
        Verify if the audio belongs to the enrolled user.
        
        :param audio: Audio signal (numpy array)
        :param sample_rate: Sample rate of the audio
        :return: (is_user: bool, confidence: float, similarity: float)
        """
        # Skip if audio is too short
        min_duration = 0.5  # seconds
        if len(audio) < sample_rate * min_duration:
            return False, 0.0, 0.0
        
        # Create embedding from incoming audio
        try:
            audio_tensor = torch.from_numpy(audio).float()
            with torch.no_grad():
                test_embedding = self.inference({
                    "waveform": audio_tensor.unsqueeze(0),
                    "sample_rate": sample_rate
                })
        except Exception as e:
            print(f"{Fore.RED}[SPEAKER] Embedding failed: {e}{Style.RESET_ALL}")
            return False, 0.0, 0.0
        
        # Calculate similarity (1 - cosine distance)
        # Convert embeddings to numpy if they're tensors
        enrolled = self.enrolled_embedding
        test = test_embedding
        
        if torch.is_tensor(enrolled):
            enrolled = enrolled.cpu().numpy()
        if torch.is_tensor(test):
            test = test.cpu().numpy()
        
        # Flatten to 1D
        enrolled = enrolled.flatten()
        test = test.flatten()
        
        # Cosine similarity (1 = identical, 0 = orthogonal, -1 = opposite)
        similarity = 1 - cosine(enrolled, test)
        
        # Verify
        is_user = similarity >= self.threshold
        confidence = min(similarity / self.threshold, 1.0)  # Normalize to 0-1
        
        # Update stats
        self.verifications_performed += 1
        self.similarity_history.append(similarity)
        
        if is_user:
            self.accepted += 1
        else:
            self.rejected += 1
        
        # Keep history limited
        if len(self.similarity_history) > 100:
            self.similarity_history.pop(0)
        
        return is_user, confidence, similarity
    
    def get_stats(self):
        """Get verification statistics."""
        avg_similarity = np.mean(self.similarity_history) if self.similarity_history else 0
        
        return {
            "verifications": self.verifications_performed,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "acceptance_rate": self.accepted / max(self.verifications_performed, 1),
            "avg_similarity": avg_similarity,
            "threshold": self.threshold
        }
    
    def adjust_threshold(self, new_threshold):
        """
        Adjust verification threshold dynamically.
        
        :param new_threshold: New threshold value (0-1)
        """
        old_threshold = self.threshold
        self.threshold = np.clip(new_threshold, 0.1, 0.95)
        
        print(f"{Fore.YELLOW}[SPEAKER] Threshold adjusted: "
              f"{old_threshold:.2f} → {self.threshold:.2f}{Style.RESET_ALL}")
    
    def print_stats(self):
        """Print verification statistics."""
        stats = self.get_stats()
        
        print(f"\n{Fore.CYAN}[SPEAKER VERIFICATION STATS]{Style.RESET_ALL}")
        print(f"  Total verifications: {stats['verifications']}")
        print(f"  Accepted: {stats['accepted']} ({stats['acceptance_rate']*100:.1f}%)")
        print(f"  Rejected: {stats['rejected']}")
        print(f"  Avg similarity: {stats['avg_similarity']:.3f}")
        print(f"  Threshold: {stats['threshold']:.3f}\n")


class AdaptiveThreshold:
    """
    Adaptively adjusts verification threshold based on performance.
    """
    
    def __init__(self, initial_threshold=0.6, target_acceptance=0.95):
        """
        :param initial_threshold: Starting threshold
        :param target_acceptance: Target acceptance rate (e.g., 0.95 = 95% of your speech accepted)
        """
        self.threshold = initial_threshold
        self.target_acceptance = target_acceptance
        self.history_size = 50
        self.recent_results = []
    
    def update(self, is_accepted):
        """Update with new verification result."""
        self.recent_results.append(1 if is_accepted else 0)
        
        # Keep only recent history
        if len(self.recent_results) > self.history_size:
            self.recent_results.pop(0)
        
        # Adjust threshold if we have enough data
        if len(self.recent_results) >= 20:
            current_acceptance = np.mean(self.recent_results)
            
            # If accepting too much, increase threshold
            if current_acceptance > self.target_acceptance + 0.05:
                self.threshold += 0.01
            # If rejecting too much, decrease threshold
            elif current_acceptance < self.target_acceptance - 0.05:
                self.threshold -= 0.01
            
            # Clamp
            self.threshold = np.clip(self.threshold, 0.3, 0.85)
    
    def get_threshold(self):
        """Get current adaptive threshold."""
        return self.threshold


# Test function
def test_speaker_verification():
    """Test speaker verification with dummy data."""
    from colorama import init
    init(autoreset=True)
    
    print(f"{Fore.CYAN}Testing Speaker Verification...{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Note: This requires an enrolled voice profile.{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Run voice_enrollment.py first if you haven't.{Style.RESET_ALL}\n")
    
    # Check if profile exists
    if not os.path.exists("user_voice_profile.pkl"):
        print(f"{Fore.RED}ERROR: No voice profile found.{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Run: python3 voice_enrollment.py{Style.RESET_ALL}")
        return
    
    try:
        verifier = SpeakerVerifier("user_voice_profile.pkl", threshold=0.6)
        
        # Test with dummy audio
        sample_rate = 16000
        duration = 2.0
        dummy_audio = np.random.randn(int(sample_rate * duration)) * 0.01
        
        print(f"{Fore.CYAN}Testing with dummy audio (will likely fail - that's expected)...{Style.RESET_ALL}")
        is_user, confidence, similarity = verifier.verify(dummy_audio, sample_rate)
        
        print(f"\n{Fore.GREEN}Results:{Style.RESET_ALL}")
        print(f"  Is User: {is_user}")
        print(f"  Confidence: {confidence:.3f}")
        print(f"  Similarity: {similarity:.3f}")
        print(f"  Threshold: {verifier.threshold:.3f}")
        
        verifier.print_stats()
        
    except Exception as e:
        print(f"{Fore.RED}Test failed: {e}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_speaker_verification()
