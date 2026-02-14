"""
JITTER DIAGNOSTIC TOOL
======================
Helps identify why the agent's movements are jittery:
1. Tests if VAE latents are stable for the same image
2. Tests if Policy predictions are stable for the same latent
3. Measures prediction variance over time
"""

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

# Import models
try:
    from models.vae import VAE
    from models.policy import TitanPolicy
except ImportError:
    import sys
    sys.path.append('.')
    from vae import VAE
    from policy import TitanPolicy

CHECKPOINT_DIR = "./checkpoints"
LATENT_DIM = 4096
TARGET_SIZE = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_models():
    """Load trained models"""
    vae = VAE(latent_dim=LATENT_DIM, img_size=TARGET_SIZE).to(device)
    policy = TitanPolicy(latent_dim=LATENT_DIM).to(device)
    
    vae_path = f"{CHECKPOINT_DIR}/vae_latest.pth"
    policy_path = f"{CHECKPOINT_DIR}/policy_latest.pth"
    
    try:
        vae.load_state_dict(torch.load(vae_path, map_location=device)['model_state_dict'])
        policy.load_state_dict(torch.load(policy_path, map_location=device))
        print("✓ Models loaded successfully")
    except Exception as e:
        print(f"✗ Error loading models: {e}")
        return None, None
    
    vae.eval()
    policy.eval()
    return vae, policy

def test_vae_stability(vae, img):
    """Test if VAE produces consistent latents for the same image"""
    print("\n=== TEST 1: VAE Stability ===")
    
    tensor = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
    
    latents = []
    with torch.no_grad():
        for i in range(100):
            mu, _ = vae.encode(tensor)
            latents.append(mu.cpu().numpy())
    
    latents = np.array(latents).squeeze()
    variance = np.var(latents, axis=0).mean()
    
    print(f"Latent variance (100 runs): {variance:.6f}")
    if variance < 0.001:
        print("✓ VAE is STABLE (low variance)")
        return True
    else:
        print("✗ VAE is UNSTABLE (high variance) - reparameterization noise?")
        return False

def test_policy_stability(policy, latent):
    """Test if Policy produces consistent predictions for the same latent"""
    print("\n=== TEST 2: Policy Stability ===")
    
    predictions = []
    with torch.no_grad():
        for i in range(100):
            pred = policy(latent)
            predictions.append(pred.cpu().numpy())
    
    predictions = np.array(predictions).squeeze()
    
    x_variance = np.var(predictions[:, 0])
    y_variance = np.var(predictions[:, 1])
    
    print(f"X prediction variance: {x_variance:.6f}")
    print(f"Y prediction variance: {y_variance:.6f}")
    
    if x_variance < 0.0001 and y_variance < 0.0001:
        print("✓ Policy is STABLE (deterministic)")
        return True
    else:
        print("✗ Policy has variance (dropout active in eval mode?)")
        return False

def test_temporal_consistency(vae, policy):
    """Simulate what happens when agent sees slightly different frames"""
    print("\n=== TEST 3: Temporal Consistency ===")
    
    # Create a simple test image (white square on black background)
    base_img = np.zeros((TARGET_SIZE, TARGET_SIZE, 3), dtype=np.uint8)
    base_img[80:180, 80:180] = 255
    
    predictions = []
    
    with torch.no_grad():
        for shift in range(-5, 6):  # Shift the square by a few pixels
            img = np.zeros((TARGET_SIZE, TARGET_SIZE, 3), dtype=np.uint8)
            img[80+shift:180+shift, 80:180] = 255
            
            tensor = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
            mu, _ = vae.encode(tensor)
            pred = policy(mu)
            
            predictions.append(pred[0, :2].cpu().numpy())
    
    predictions = np.array(predictions)
    
    # Check if predictions change smoothly or jump erratically
    deltas = np.abs(np.diff(predictions, axis=0))
    max_jump = np.max(deltas)
    avg_jump = np.mean(deltas)
    
    print(f"Max jump between adjacent frames: {max_jump:.4f}")
    print(f"Avg jump between adjacent frames: {avg_jump:.4f}")
    
    if max_jump > 0.1:
        print("✗ Policy is SENSITIVE to small visual changes (causes jitter)")
        return False
    else:
        print("✓ Policy predictions change smoothly")
        return True

def visualize_prediction_distribution(policy, vae):
    """Show where the policy tends to predict"""
    print("\n=== TEST 4: Prediction Distribution ===")
    
    # Load some random training data
    try:
        import glob
        memory_files = glob.glob("./data/memories/*.npz")
        if not memory_files:
            print("No training data found, skipping...")
            return
        
        # Sample 1000 random latents
        all_latents = []
        for f in memory_files[:10]:
            data = np.load(f)
            latents = data['latents']
            indices = np.random.choice(len(latents), min(100, len(latents)), replace=False)
            all_latents.append(latents[indices])
        
        all_latents = np.concatenate(all_latents, axis=0)
        
        # Get predictions
        predictions = []
        with torch.no_grad():
            for latent in all_latents:
                latent_tensor = torch.from_numpy(latent).float().unsqueeze(0).to(device)
                pred = policy(latent_tensor)
                predictions.append(pred[0, :2].cpu().numpy())
        
        predictions = np.array(predictions)
        
        # Plot heatmap
        plt.figure(figsize=(10, 8))
        plt.hexbin(predictions[:, 0], predictions[:, 1], gridsize=30, cmap='hot')
        plt.colorbar(label='Prediction Density')
        plt.xlabel('X Prediction (0=left, 1=right)')
        plt.ylabel('Y Prediction (0=top, 1=bottom)')
        plt.title('Policy Prediction Distribution\n(Should be spread across screen, not clustered)')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.savefig('prediction_heatmap.png', dpi=150)
        print("✓ Saved prediction_heatmap.png")
        
        # Check if predictions are clustered (bad) or spread out (good)
        x_std = np.std(predictions[:, 0])
        y_std = np.std(predictions[:, 1])
        
        print(f"X spread (std): {x_std:.3f}")
        print(f"Y spread (std): {y_std:.3f}")
        
        if x_std < 0.1 or y_std < 0.1:
            print("✗ Predictions are CLUSTERED (model not trained enough)")
            return False
        else:
            print("✓ Predictions are SPREAD OUT")
            return True
            
    except Exception as e:
        print(f"Error: {e}")
        return None

def main():
    print("=" * 60)
    print("JITTER DIAGNOSTIC TOOL")
    print("=" * 60)
    
    vae, policy = load_models()
    if vae is None or policy is None:
        return
    
    # Create a test image (screenshot if available, else synthetic)
    if Path("agent_view.jpg").exists():
        img = cv2.imread("agent_view.jpg")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (TARGET_SIZE, TARGET_SIZE))
        print(f"Using agent_view.jpg for testing")
    else:
        img = np.random.randint(0, 255, (TARGET_SIZE, TARGET_SIZE, 3), dtype=np.uint8)
        print("No agent_view.jpg found, using random noise")
    
    # Run diagnostics
    with torch.no_grad():
        tensor = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
        mu, _ = vae.encode(tensor)
        
        vae_stable = test_vae_stability(vae, img)
        policy_stable = test_policy_stability(policy, mu)
        temporal_stable = test_temporal_consistency(vae, policy)
        spread_ok = visualize_prediction_distribution(policy, vae)
    
    # Summary
    print("\n" + "=" * 60)
    print("DIAGNOSIS SUMMARY")
    print("=" * 60)
    
    issues = []
    if not vae_stable:
        issues.append("⚠ VAE produces noisy latents (reparameterization in eval mode?)")
    if not policy_stable:
        issues.append("⚠ Policy predictions vary (dropout in eval mode?)")
    if not temporal_stable:
        issues.append("⚠ Policy overreacts to small visual changes (undertrained)")
    if spread_ok == False:
        issues.append("⚠ Policy predictions clustered (needs more training data)")
    
    if not issues:
        print("✓ All tests passed! Jitter is likely due to:")
        print("  - UPDATE_RATE too high (try 3-5 Hz)")
        print("  - MOVE_SPEED too low (try 0.2-0.3)")
        print("  - EMA_ALPHA too high (try 0.2-0.4)")
    else:
        print("Issues found:")
        for issue in issues:
            print(f"  {issue}")
    
    print("\nRecommendations:")
    print("  1. Check prediction_heatmap.png - should cover full screen")
    print("  2. If clustered → train policy longer")
    print("  3. If VAE unstable → set vae.eval() and disable reparameterization")
    print("  4. If policy unstable → check for dropout in eval mode")

if __name__ == "__main__":
    main()
