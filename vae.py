import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# 1. BLOC RÉSIDUEL (Architecture Skip-Connection)
# =============================================================================
class ResBlock(nn.Module):
    """
    Bloc fondamental permettant au gradient de traverser le réseau sans s'atténuer.
    Architecture : Conv -> LeakyReLU -> Conv -> Somme (Skip Connection) -> LeakyReLU
    """
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return F.leaky_relu(x + self.conv(x), 0.2)

# =============================================================================
# 2. ARCHITECTURE VAE TITAN (COMPLÈTE)
# =============================================================================
class VAE(nn.Module):
    def __init__(self, latent_dim=2048, img_size=128):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.img_size = img_size
        
        # --- ENCODER (Compression: 128x128 -> 8x8) ---
        self.encoder = nn.Sequential(
            # Layer 1: 128 -> 64
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 2: 64 -> 32
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(128),
            
            # Layer 3: 32 -> 16
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(256),
            
            # Layer 4: 16 -> 8
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(512)
        )
        
        # Calcul de la taille à plat (Flatten)
        self.flat_size = 512 * 8 * 8
        
        # Projection vers l'espace latent (Cerveau)
        self.fc_mu = nn.Linear(self.flat_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_size, latent_dim)
        
        # Projection inverse
        self.fc_decode = nn.Linear(latent_dim, self.flat_size)
        
        # --- DECODER (Decompression: 8x8 -> 128x128) ---
        self.decoder = nn.Sequential(
            # Layer 1: 8 -> 16
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(256),
            
            # Layer 2: 16 -> 32
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(128),
            
            # Layer 3: 32 -> 64
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(64),
            
            # Layer 4: 64 -> 128
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            # Sigmoid force la sortie entre 0.0 et 1.0 (RGB)
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1) # Flatten
        mu = self.fc_mu(h)
        # On clamp logvar pour éviter les NaN lors du calcul de exp()
        logvar = torch.clamp(self.fc_logvar(h), min=-6, max=6)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_decode(z)
        h = h.view(h.size(0), 512, 8, 8) # Unflatten
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

# =============================================================================
# 3. MOTEUR DE PERTE (LOSS) STABILISÉ
# =============================================================================
class Laplacian(nn.Module):
    """Filtre de détection de contours pour forcer la netteté du texte."""
    def __init__(self, device):
        super(Laplacian, self).__init__()
        # Noyau Laplacien standard
        kernel = torch.tensor([
            [0,  1, 0],
            [1, -4, 1],
            [0,  1, 0]
        ], dtype=torch.float32).to(device)
        # Adaptation aux dimensions (Channels, Output, H, W)
        self.kernel = kernel.view(1, 1, 3, 3).repeat(3, 1, 1, 1)

    def forward(self, x):
        # Convolution groupée (traite chaque canal RGB indépendamment)
        return F.conv2d(x, self.kernel, padding=1, groups=3)

# Instance globale pour éviter de recréer le kernel à chaque itération
_laplacian_engine = None

def vae_loss_function(recon, x, mu, logvar, beta, edge_weight, chroma_weight):
    global _laplacian_engine
    if _laplacian_engine is None:
        _laplacian_engine = Laplacian(x.device)

    # 1. Double Reconstruction : L1 pour la fidélité, MSE pour le contraste (Anti-Gris)
    recon_l1 = F.l1_loss(recon, x, reduction='mean')
    recon_mse = F.mse_loss(recon, x, reduction='mean')
    
    # On booste la reconstruction de base (Poids total 40 au lieu de 10)
    total_recon_loss = (recon_l1 * 20.0) + (recon_mse * 20.0)
    
    # 2. Perte Laplacienne (Netteté)
    lap_loss = 0.0
    if edge_weight > 0:
        lap_recon = _laplacian_engine(recon)
        lap_real = _laplacian_engine(x)
        # On utilise MSE ici aussi pour que les bords soient "violents" pour le réseau
        lap_loss = F.mse_loss(lap_recon, lap_real, reduction='mean') * edge_weight
    
    # 3. KL Divergence (On la garde très basse pour l'instant)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp() + 1e-6)
    
    return total_recon_loss + lap_loss + (beta * kl_loss)