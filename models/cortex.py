import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_head, max_len):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.head_dim = d_model // n_head
        
        self.c_attn = nn.Linear(d_model, 3 * d_model)
        self.c_proj = nn.Linear(d_model, d_model)
        
        # Causal Mask (Tril) - Ensures we can't see the future
        self.register_buffer("bias", torch.tril(torch.ones(max_len, max_len))
                                     .view(1, 1, max_len, max_len))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(C, dim=2)
        
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Flash Attention Logic (Manual implementation for compatibility)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class CortexBlock(nn.Module):
    def __init__(self, d_model, n_head, max_len):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, max_len)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class TitanCortex(nn.Module):
    def __init__(self, state_dim=4096, action_dim=6, d_model=1024, n_layer=8, n_head=8, max_len=128):
        super().__init__()
        self.max_len = max_len
        
        # Projections
        self.state_emb = nn.Linear(state_dim, d_model)
        self.action_emb = nn.Linear(action_dim, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        
        self.blocks = nn.Sequential(*[
            CortexBlock(d_model, n_head, max_len) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, state_dim)

    def forward(self, states, actions):
        # states: [Batch, Seq, 4096]
        # actions: [Batch, Seq, 6]
        B, T, _ = states.size()
        
        # Fuse State + Action
        # "I am seeing X and doing Y"
        token_embeddings = self.state_emb(states) + self.action_emb(actions)
        position_embeddings = self.pos_emb(torch.arange(T, device=states.device))
        
        x = token_embeddings + position_embeddings
        x = self.blocks(x)
        x = self.ln_f(x)
        return self.head(x)