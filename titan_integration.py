#!/usr/bin/env python3
"""
TITAN Backend Integration
Connects the UI to VAE, Policy, Cortex, and Voice systems
"""

import os
import sys
import numpy as np
import cv2
import mss
import torch
import threading
import time
from collections import deque

# Add models to path
sys.path.insert(0, os.path.join(os.getcwd(), 'models'))

class TitanBackend:
    """
    Backend manager for TITAN agent systems.
    Handles vision, policy, cortex, and voice integration.
    """
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  [BACKEND] Using device: {self.device}")
        
        # Model paths
        self.checkpoint_dir = "./checkpoints"
        
        # Models (lazy loading)
        self.vae = None
        self.policy = None
        self.cortex = None
        
        # State
        self.vision_active = False
        self.agent_active = False
        self.voice_active = False
        
        # Screen capture
        self.sct = mss.mss()
        self.target_size = 256
        
        # Performance tracking
        self.fps_counter = deque(maxlen=30)
        self.last_frame_time = time.time()
        
        # Callbacks
        self.on_vision_update = None
        self.on_action_update = None
        self.on_voice_detected = None
        
        print("  [BACKEND] TITAN Backend initialized")
    
    def load_models(self):
        """Load neural network models"""
        print("  [BACKEND] Loading models...")
        
        try:
            # Load VAE
            if self._model_exists("vae_latest.pth"):
                from vae import VAE
                self.vae = VAE(latent_dim=4096, img_size=256).to(self.device)
                self._load_weights(self.vae, "vae_latest.pth")
                self.vae.eval()
                print("    ✓ VAE loaded")
            else:
                print("    ⚠ VAE checkpoint not found")
            
            # Load Policy
            if self._model_exists("policy_latest.pth"):
                from policy import TitanPolicy
                self.policy = TitanPolicy(latent_dim=4096).to(self.device)
                self._load_weights(self.policy, "policy_latest.pth")
                self.policy.eval()
                print("    ✓ Policy loaded")
            else:
                print("    ⚠ Policy checkpoint not found")
            
            # Load Cortex
            if self._model_exists("cortex_latest.pth"):
                from cortex import TitanCortex
                self.cortex = TitanCortex(state_dim=4096).to(self.device)
                self._load_weights(self.cortex, "cortex_latest.pth")
                self.cortex.eval()
                print("    ✓ Cortex loaded")
            else:
                print("    ⚠ Cortex checkpoint not found")
            
            return True
            
        except Exception as e:
            print(f"  [ERROR] Model loading failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _model_exists(self, filename):
        """Check if model checkpoint exists"""
        return os.path.exists(os.path.join(self.checkpoint_dir, filename))
    
    def _load_weights(self, model, filename):
        """Load model weights from checkpoint"""
        path = os.path.join(self.checkpoint_dir, filename)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    
    def capture_screen(self):
        """Capture current screen and return as numpy array"""
        try:
            # Capture primary monitor
            monitor = self.sct.monitors[1]
            sct_img = self.sct.grab(monitor)
            
            # Convert to numpy array
            img = np.array(sct_img)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            
            # Resize to target size
            img = cv2.resize(img, (self.target_size, self.target_size), 
                           interpolation=cv2.INTER_AREA)
            
            return img
            
        except Exception as e:
            print(f"  [ERROR] Screen capture failed: {e}")
            return None
    
    def encode_frame(self, frame):
        """Encode frame to latent space using VAE"""
        if self.vae is None:
            return None
        
        try:
            # Convert to tensor
            tensor = torch.from_numpy(frame).float().permute(2, 0, 1) / 255.0
            tensor = tensor.unsqueeze(0).to(self.device)
            
            # Encode
            with torch.no_grad():
                mu, _ = self.vae.encode(tensor)
            
            return mu
            
        except Exception as e:
            print(f"  [ERROR] Frame encoding failed: {e}")
            return None
    
    def predict_action(self, latent):
        """Predict action from latent representation"""
        if self.policy is None or latent is None:
            return None
        
        try:
            with torch.no_grad():
                action = self.policy(latent)
            
            return action.cpu().numpy()[0]
            
        except Exception as e:
            print(f"  [ERROR] Action prediction failed: {e}")
            return None
    
    def vision_loop(self):
        """Main vision processing loop"""
        print("  [VISION] Vision loop started")
        
        while self.vision_active:
            start_time = time.time()
            
            # Capture frame
            frame = self.capture_screen()
            if frame is None:
                time.sleep(0.1)
                continue
            
            # Encode to latent
            latent = self.encode_frame(frame)
            
            # Predict action if agent is active
            action = None
            if self.agent_active and latent is not None:
                action = self.predict_action(latent)
            
            # Calculate FPS
            elapsed = time.time() - start_time
            fps = 1.0 / elapsed if elapsed > 0 else 0
            self.fps_counter.append(fps)
            
            # Callbacks
            if self.on_vision_update:
                avg_fps = np.mean(self.fps_counter)
                self.on_vision_update(frame, latent, avg_fps)
            
            if action is not None and self.on_action_update:
                self.on_action_update(action)
            
            # Rate limiting (don't go too fast)
            time.sleep(max(0, 0.1 - elapsed))
        
        print("  [VISION] Vision loop stopped")
    
    def start_vision(self):
        """Start vision system"""
        if not self.vision_active:
            self.vision_active = True
            threading.Thread(target=self.vision_loop, daemon=True).start()
    
    def stop_vision(self):
        """Stop vision system"""
        self.vision_active = False
    
    def start_agent(self):
        """Start autonomous agent"""
        self.agent_active = True
        print("  [AGENT] Agent activated")
    
    def stop_agent(self):
        """Stop autonomous agent"""
        self.agent_active = False
        print("  [AGENT] Agent deactivated")
    
    def get_model_info(self):
        """Get information about loaded models"""
        info = {
            'vae': self.vae is not None,
            'policy': self.policy is not None,
            'cortex': self.cortex is not None,
            'device': str(self.device)
        }
        return info
    
    def cleanup(self):
        """Cleanup resources"""
        self.stop_vision()
        self.stop_agent()
        
        if hasattr(self, 'sct'):
            try:
                self.sct.close()
            except:
                pass


class VoiceSystem:
    """
    Voice system integration for TITAN.
    Handles speaker verification, wake word detection, and AEC.
    """
    
    def __init__(self):
        self.active = False
        self.speaker_verified = False
        
        # Import voice modules if available
        try:
            from speaker_verification import SpeakerVerifier
            
            # Check if profile exists
            if os.path.exists("user_voice_profile.pkl"):
                self.verifier = SpeakerVerifier("user_voice_profile.pkl")
                print("  [VOICE] Speaker verification loaded")
            else:
                self.verifier = None
                print("  [VOICE] No voice profile found - run voice_enrollment.py")
        except Exception as e:
            print(f"  [VOICE] Voice system unavailable: {e}")
            self.verifier = None
    
    def start(self):
        """Start voice system"""
        self.active = True
        print("  [VOICE] Voice system activated")
    
    def stop(self):
        """Stop voice system"""
        self.active = False
        print("  [VOICE] Voice system deactivated")
    
    def verify_speaker(self, audio):
        """Verify if audio is from enrolled user"""
        if self.verifier is None:
            return False, 0.0, 0.0
        
        return self.verifier.verify(audio)


# Convenience function for UI
def create_backend():
    """Create and initialize TITAN backend"""
    backend = TitanBackend()
    
    # Try to load models
    backend.load_models()
    
    return backend
