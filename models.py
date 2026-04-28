import math
import torch
from torch import nn

class ResidualBlock(nn.Module):
    def __init__(self, main, skip=None):
        super().__init__()
        self.main = nn.Sequential(*main)
        self.skip = skip if skip else nn.Identity()

    def forward(self, input):
        return self.main(input) + self.skip(input)


class ResConvBlock(ResidualBlock):
    def __init__(self, c_in, c_mid, c_out, num_groups = 16, last_layer = False):
        skip = None if c_in == c_out else nn.Conv2d(c_in, c_out, 1, bias=False)
        super().__init__([
            nn.Conv2d(c_in, c_mid, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups, c_mid),
            nn.SiLU(inplace=True),
            nn.Conv2d(c_mid, c_out, 3, padding=1, bias= not last_layer),
            nn.GroupNorm(num_groups, c_out) if not last_layer else nn.Identity(),
            nn.SiLU(inplace=True) if not last_layer else nn.Identity(),
        ], skip)


class SkipBlock(nn.Module):
    def __init__(self, *main, skip=None):
        super().__init__()
        self.main = nn.Sequential(*main)
        self.skip = skip if skip else nn.Identity()

    def forward(self, input):
        return torch.cat([self.main(input), self.skip(input)], dim=1)

class SinusoidalTimestepEmbedder(nn.Module):
    def __init__(self, embedding_dim, max_period=10000):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_period = max_period
        
    def forward(self, t):
        half_dim = self.embedding_dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=t.device) / half_dim
        )
        args = t[:, None].float() * freqs[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.embedding_dim % 2 == 1:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding


class TimestepEmbedder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim_mult=4):
        super().__init__()
        self.sinusoidal = SinusoidalTimestepEmbedder(embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * hidden_dim_mult),
            nn.SiLU(),
            nn.Linear(embedding_dim * hidden_dim_mult, embedding_dim)
        )
        
    def forward(self, t):
        emb = self.sinusoidal(t)
        return self.mlp(emb)


class PositionalEncoding2d(nn.Module):
    def __init__(self, embedding_dim, frequency=10000):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.frequency = frequency
        # cache position embeddings for efficiency
        self.register_buffer("pos_embed", None, persistent=False)
        
    def forward(self, x):
        B, C, H, W = x.shape
        device = x.device
        
        if self.pos_embed is None or self.pos_embed.shape[-2:] != (H, W):
            half_dim = self.embedding_dim // 2
            
            pos_y = torch.arange(H, device=device).reshape(H, 1)
            pos_x = torch.arange(W, device=device).reshape(W, 1)
            
            div_term = torch.exp(torch.arange(0, half_dim, 2, device=device) * 
                                -(math.log(self.frequency) / half_dim))
            
            pe_y = torch.zeros(H, half_dim, device=device)
            pe_y[:, 0::2] = torch.sin(pos_y * div_term)
            pe_y[:, 1::2] = torch.cos(pos_y * div_term)
            
            pe_x = torch.zeros(W, half_dim, device=device)
            pe_x[:, 0::2] = torch.sin(pos_x * div_term)
            pe_x[:, 1::2] = torch.cos(pos_x * div_term)
            
            pe = torch.zeros(H, W, self.embedding_dim, device=device)
            pe[:, :, :half_dim] = pe_y[:, None, :]
            pe[:, :, half_dim:] = pe_x[None, :, :]
            
            self.pos_embed = pe.reshape(1, H * W, self.embedding_dim)
        
        x_flat = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        x_flat = x_flat + self.pos_embed.expand(B, -1, -1)
        
        return x_flat.reshape(B, H, W, C).permute(0, 3, 1, 2)

class SelfAttention(nn.Module):
    def __init__(self, channels, heads=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels * 4, bias=True),
            nn.GELU(),
            nn.Linear(channels * 4, channels, bias=True),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, -1).permute(0, 2, 1)  # (B, H*W, C)
        x = self.norm1(x)
        x = x + self.attention(x, x, x)[0]
        x = x + self.mlp(self.norm2(x))
        return x.permute(0, 2, 1).view(B, C, H, W)
