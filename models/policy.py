import torch
import torch.nn as nn

class TitanPolicy(nn.Module):
    def __init__(self, latent_dim=4096, action_dim=6):
        super().__init__()
        
        # 1. The Core Brain (Process the concept)
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Dropout(0.2),
            
            nn.Linear(2048, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.2),
            
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU()
        )
        
        # HEAD 1: Mouse Coordinates (X, Y)
        # KEEP SIGMOID: We need strict 0.0 to 1.0 range for the screen.
        self.head_mouse = nn.Sequential(
            nn.Linear(512, 2),
            nn.Sigmoid() 
        )
        
        # HEAD 2: Clicks (Left, Right, Scroll, Key)
        # REMOVE SIGMOID: We will output "Logits" (raw scores) for stability.
        self.head_aux = nn.Sequential(
            nn.Linear(512, 4)
        )

    def forward(self, x):
        # x: (Batch, 4096)
        features = self.net(x)
        
        mouse = self.head_mouse(features) # (B, 2)
        aux = self.head_aux(features)     # (B, 4)
        
        # --- THE FIX IS HERE ---
        # We concatenate them into a single tensor of shape (B, 6)
        return torch.cat([mouse, aux], dim=1)