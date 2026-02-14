import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# =============================================================================
# HELPER: LAPLACIAN FILTER (For Edge Loss)
# =============================================================================
def laplacian_filter(img):
    kernel = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=torch.float32)
    kernel = kernel.view(1, 1, 3, 3).repeat(3, 1, 1, 1).to(img.device)
    return F.conv2d(img, kernel, padding=1, groups=3)

# =============================================================================
# 1. PERCEPTUAL LOSS ENGINE (VGG19)
# =============================================================================
class VGGLoss(nn.Module):
    def __init__(self, device):
        super(VGGLoss, self).__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        
        for x in range(2): self.slice1.add_module(str(x), vgg[x])
        for x in range(2, 7): self.slice2.add_module(str(x), vgg[x])
        for x in range(7, 12): self.slice3.add_module(str(x), vgg[x])
        
        self.slice1.to(device)
        self.slice2.to(device)
        self.slice3.to(device)
            
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        # SAFETY: Clamp inputs to prevent VGG from seeing Infinity
        x = torch.clamp(x, 0, 1)
        y = torch.clamp(y, 0, 1)
        
        # Normalize (ImageNet stats)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(x.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(x.device)
        
        x = (x - mean) / std
        y = (y - mean) / std

        h_x1 = self.slice1(x)
        h_x2 = self.slice2(h_x1)
        h_x3 = self.slice3(h_x2)

        h_y1 = self.slice1(y)
        h_y2 = self.slice2(h_y1)
        h_y3 = self.slice3(h_y2)

        loss = F.l1_loss(h_x1, h_y1) * 1.0 + \
               F.l1_loss(h_x2, h_y2) * 1.0 + \
               F.l1_loss(h_x3, h_y3) * 0.5
        return loss

# =============================================================================
# 2. RESIDUAL BLOCK (GROUP NORM EDITION)
# =============================================================================
class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            # FIX: GroupNorm instead of BatchNorm (Safe for BatchSize=4)
            nn.GroupNorm(8, channels), 
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, channels)
        )

    def forward(self, x):
        return F.leaky_relu(x + self.conv(x), 0.2)

# =============================================================================
# 3. TITAN VAE ARCHITECTURE (V4.1 STABLE)
# =============================================================================
class VAE(nn.Module):
    def __init__(self, latent_dim=4096, img_size=256):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.final_feat_size = img_size // 32 
        
        # --- ENCODER ---
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True), ResBlock(64),
            nn.Conv2d(64, 128, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True), ResBlock(128),
            nn.Conv2d(128, 256, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True), ResBlock(256),
            nn.Conv2d(256, 256, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True), ResBlock(256)
        )
        
        self.flat_size = 256 * self.final_feat_size * self.final_feat_size 
        
        self.fc_mu = nn.Linear(self.flat_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_size, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, self.flat_size)
        
        # --- DECODER (With GroupNorm) ---
        self.decoder_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(256, 256, 3, 1, 1), nn.GroupNorm(8, 256), nn.LeakyReLU(0.2, inplace=True), ResBlock(256),
            
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(256, 128, 3, 1, 1), nn.GroupNorm(8, 128), nn.LeakyReLU(0.2, inplace=True), ResBlock(128),
            
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 64, 3, 1, 1), nn.GroupNorm(8, 64), nn.LeakyReLU(0.2, inplace=True), ResBlock(64),
            
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 32, 3, 1, 1), nn.GroupNorm(8, 32), nn.LeakyReLU(0.2, inplace=True),
            
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(32, 16, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(16, 3, 3, 1, 1),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder_conv(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        # SAFETY CLAMP
        logvar = torch.clamp(self.fc_logvar(h), min=-6.0, max=2.0)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_decode(z)
        h = h.view(-1, 256, self.final_feat_size, self.final_feat_size)
        return self.decoder_conv(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

# =============================================================================
# 4. UPDATED VAE LOSS FUNCTION (V5.1 - SIGNAL HAMMER EDITION)
# =============================================================================
_vgg_loss_engine = None

def vae_loss_function(recon, x, mu, logvar, beta, mse_weight, edge_weight, chroma_weight, perceptual_weight):
    global _vgg_loss_engine
    
    # --- 1. ENFORCE HIGH PRECISION ---
    # We disable autocast here because extreme weights (100+) can overflow 
    # half-precision (Float16) and cause NaNs.
    with torch.amp.autocast('cuda', enabled=False):
        recon = recon.float()
        x = x.float()
        mu = mu.float()
        logvar = logvar.float()

        # --- 2. RECONSTRUCTION LOSS (The "DC Signal") ---
        # Switch to MSE: Punishes large color gaps (Red vs Grey) much harder than SmoothL1.
        recon_loss = F.mse_loss(recon, x, reduction='mean') * mse_weight
        
        # --- 3. EDGE LOSS (High-Frequency Text/Lines) ---
        edge_loss = torch.tensor(0.0, device=x.device)
        if edge_weight > 0:
            # Laplacian highlights text borders and code lines
            edge_loss = F.mse_loss(laplacian_filter(recon), laplacian_filter(x)) * edge_weight
        
        # --- 4. CHROMA LOSS (Version Dark-Mode Friendly) ---
        chroma_loss = torch.tensor(0.0, device=x.device)
        if chroma_weight > 0:
            def get_chroma(img): return img - img.mean(dim=1, keepdim=True)
            # ON PASSE EN L1 ICI (F.l1_loss au lieu de F.mse_loss)
            chroma_loss = F.l1_loss(get_chroma(recon), get_chroma(x)) * chroma_weight
        
        # --- 5. PERCEPTUAL LOSS (Structural Brain) ---
        perc_loss = torch.tensor(0.0, device=x.device)
        if perceptual_weight > 0:
            if _vgg_loss_engine is None: 
                _vgg_loss_engine = VGGLoss(x.device)
            perc_loss = _vgg_loss_engine(recon, x) * perceptual_weight

        # --- 6. KL DIVERGENCE (The "Compressor") ---
        # Safety: We clamp logvar and mu to prevent the latent space from exploding 
        # during the 'Signal Hammer' phase.
        mu_clamped = torch.clamp(mu, -10, 10)
        logvar_clamped = torch.clamp(logvar, -10, 10)
        
        # Standard VAE KL formula
        kl_loss = -0.5 * torch.mean(1 + logvar_clamped - mu_clamped.pow(2) - logvar_clamped.exp())
        
        # Final Summation
        total_loss = recon_loss + edge_loss + chroma_loss + perc_loss + (beta * kl_loss)

        # --- 7. EMERGENCY DEBUG CATCHER ---
        if torch.isnan(total_loss):
            print(f"\n[!!!] NAN DETECTED IN LOSS COMPONENT")
            print(f"MSE: {recon_loss.item():.2f} | Chroma: {chroma_loss.item():.2f}")
            print(f"Edge: {edge_loss.item():.2f} | KL: {kl_loss.item():.2f} (Beta: {beta})")
            # We return a very high loss instead of NaN to prevent optimizer crash
            return torch.tensor(1000.0, device=x.device, requires_grad=True)

        return total_loss