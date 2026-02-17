import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# =============================================================================
# 1. PERCEPTUAL LOSS ENGINE (VGG19)
# =============================================================================
class VGGLoss(nn.Module):
    def __init__(self, device):
        super(VGGLoss, self).__init__()
        # Load VGG19, slice it to get features from specific layers
        # We use VGG19 features to determine "perceptual" distance (texture/shape)
        # rather than just pixel color distance.
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        
        # Slice 1: Relu_1_2
        for x in range(2): 
            self.slice1.add_module(str(x), vgg[x])
            
        # Slice 2: Relu_2_2
        for x in range(2, 7): 
            self.slice2.add_module(str(x), vgg[x])
            
        # Slice 3: Relu_3_2
        for x in range(7, 12): 
            self.slice3.add_module(str(x), vgg[x])
        
        self.slice1.to(device)
        self.slice2.to(device)
        self.slice3.to(device)
        
        # Freeze VGG weights (we don't train this, we just use it as a ruler)
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        # Clamp inputs to [0,1] for safety
        x = torch.clamp(x, 0, 1)
        y = torch.clamp(y, 0, 1)
        
        # Normalize for VGG (ImageNet stats)
        # This is critical for the pre-trained VGG to "see" the image correctly
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1,3,1,1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1,3,1,1)
        
        x = (x - mean) / std
        y = (y - mean) / std

        # Compute features at different depths
        h_x1 = self.slice1(x)
        h_y1 = self.slice1(y)
        
        h_x2 = self.slice2(h_x1)
        h_y2 = self.slice2(h_y1)
        
        h_x3 = self.slice3(h_x2)
        h_y3 = self.slice3(h_y2)

        # L1 Loss between features (Perceptual Distance)
        loss = F.l1_loss(h_x1, h_y1) + F.l1_loss(h_x2, h_y2) + F.l1_loss(h_x3, h_y3)
        return loss

# =============================================================================
# 2. BUILDING BLOCKS (RESNET + PIXELSHUFFLE)
# =============================================================================
class ResBlock(nn.Module):
    """
    Standard Residual Block with GroupNorm.
    GroupNorm is used instead of BatchNorm because our batch size is small.
    """
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, channels)
        )

    def forward(self, x):
        return F.leaky_relu(x + self.conv(x), 0.2)

class SpatialUpsample(nn.Module):
    """
    Upscales image by 2x using PixelShuffle (Sub-Pixel Convolution).
    This creates sharper edges for text than standard interpolation.
    """
    def __init__(self, in_c, out_c):
        super().__init__()
        # PixelShuffle x2 requires 4x output channels
        self.conv = nn.Conv2d(in_c, out_c * 4, kernel_size=3, padding=1)
        self.shuff = nn.PixelShuffle(2)
        self.norm = nn.GroupNorm(8, out_c)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        # Refine the upsampled result with a ResBlock
        self.res = ResBlock(out_c)
        
    def forward(self, x):
        x = self.act(self.norm(self.shuff(self.conv(x))))
        return self.res(x)

# =============================================================================
# 3. MAIN ARCHITECTURE: SPATIAL VAE (TITAN V5)
# =============================================================================
class VAE(nn.Module):
    def __init__(self, latent_dim=None, img_size=256):
        super(VAE, self).__init__()
        
        # --- ENCODER (Downsampling) ---
        # 256x256 -> 128x128
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1), 
            nn.LeakyReLU(0.2), 
            ResBlock(64)
        )
        
        # 128x128 -> 64x64
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1), 
            nn.GroupNorm(8, 128), 
            nn.LeakyReLU(0.2), 
            ResBlock(128)
        )
        
        # 64x64 -> 32x32
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, 1), 
            nn.GroupNorm(8, 256), 
            nn.LeakyReLU(0.2), 
            ResBlock(256), 
            ResBlock(256) # Extra depth at the bottleneck
        )
        
        # --- SPATIAL LATENT SPACE ---
        # We do NOT flatten. We keep the layout 32x32x32.
        self.latent_channels = 32
        self.mu_conv = nn.Conv2d(256, self.latent_channels, 3, 1, 1)
        self.logvar_conv = nn.Conv2d(256, self.latent_channels, 3, 1, 1)
        
        # --- DECODER (Upsampling) ---
        self.dec_input = nn.Conv2d(self.latent_channels, 256, 3, 1, 1)
        
        self.up1 = SpatialUpsample(256, 128) # 32 -> 64
        self.up2 = SpatialUpsample(128, 64)  # 64 -> 128
        self.up3 = SpatialUpsample(64, 32)   # 128 -> 256
        
        self.final = nn.Sequential(
            ResBlock(32),
            nn.Conv2d(32, 3, 3, 1, 1),
            nn.Sigmoid() # Output 0-1 range
        )

    def encode(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        
        mu = self.mu_conv(x)
        logvar = self.logvar_conv(x)
        
        # --- SAFETY CLAMP (CRITICAL FOR NAN PREVENTION) ---
        # We clamp log-variance to prevent exp() from creating Infinity.
        logvar = torch.clamp(logvar, min=-10, max=10)
        
        return mu, logvar

    def reparameterize(self, mu, logvar):
        # Standard VAE reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.dec_input(z)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.final(x)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

# =============================================================================
# 4. ROBUST LOSS FUNCTION (NAN SAFE + SHARPNESS)
# =============================================================================
# Global variable to hold the VGG model so we don't reload it every batch
_vgg_loss_engine = None

def compute_gradient_loss(input, target):
    """
    Computes the first-order derivatives (edges) and penalizes mismatch.
    This creates sharp text by forcing the 'change' in pixel values to match.
    """
    # Horizontal Gradients (dx)
    # We slice to ignore the last column of input and first of target to align
    dx_input = input[:, :, :, 1:] - input[:, :, :, :-1]
    dx_target = target[:, :, :, 1:] - target[:, :, :, :-1]
    loss_dx = F.l1_loss(dx_input, dx_target)

    # Vertical Gradients (dy)
    dy_input = input[:, :, 1:, :] - input[:, :, :-1, :]
    dy_target = target[:, :, 1:, :] - target[:, :, :-1, :]
    loss_dy = F.l1_loss(dy_input, dy_target)

    return loss_dx + loss_dy

def vae_loss_function(recon, x, mu, logvar, beta, mse_weight, edge_weight, chroma_weight, perceptual_weight):
    global _vgg_loss_engine
    
    # 1. PIXEL LOSS (Structure)
    # Using L1 instead of MSE creates sharper results for text.
    pixel_loss = F.l1_loss(recon, x, reduction='mean') * mse_weight
    
    # 2. EDGE LOSS (Readability)
    # This specifically penalizes blurry edges on text.
    edge_loss = torch.tensor(0.0, device=x.device)
    if edge_weight > 0:
        edge_loss = compute_gradient_loss(recon, x) * edge_weight
    
    # 3. PERCEPTUAL LOSS (Texture)
    # Uses VGG19 to match the "style" or "features" of the image
    perc_loss = torch.tensor(0.0, device=x.device)
    if perceptual_weight > 0:
        if _vgg_loss_engine is None: 
            # Instantiate the VGGLoss class defined above
            _vgg_loss_engine = VGGLoss(x.device)
        perc_loss = _vgg_loss_engine(recon, x) * perceptual_weight

    # 4. KL DIVERGENCE (Compression)
    # We sum over channels (dim=1) and average over spatial dimensions (dim=2,3).
    kl_element = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kl_loss = torch.sum(kl_element, dim=1) # Sum channels
    kl_loss = torch.mean(kl_loss)          # Mean over Batch, H, W

    # 5. FINAL SUMMATION
    total_loss = pixel_loss + edge_loss + perc_loss + (beta * kl_loss)
    
    # Emergency NaN Catcher
    if torch.isnan(total_loss):
        print(" [!] VAE LOSS NAN DETECTED")
        return torch.tensor(1000.0, device=x.device, requires_grad=True)
        
    return total_loss