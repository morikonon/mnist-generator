import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalTimeEmbedding(nn.Module):
  """Embeds a continuous time value t in [0, 1] into a high-dimensional vector."""
  def __init__(self, dim: int):
    super().__init__()

    self.dim = dim

  def forward(self, t: torch.Tensor) -> torch.Tensor:

    half_dim = self.dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
    emb = t[:, None] * emb[None, :]
    emb = torch.cat((emb.sin(), emb.cos()), dim=-1)

    return emb
  
class AdaLN(nn.Module):
  """Adaptive Layer Normalization to inject time embedding."""
  def __init__(self, d_model: int):
    super().__init__()

    self.norm = nn.LayerNorm(d_model)

  def forward(self, x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    return self.norm(x) * (1 + scale) + shift
  
class DiTBlock(nn.Module):
  """A single Diffusion Transformer Block with AdaLN conditioning."""
  def __init__(self, d_model: int, n_heads: int):
    super().__init__()

    self.d_model = d_model
    self.n_heads = n_heads

    # Adaptive Layer Normalization
    self.ada_ln1 = AdaLN(d_model)
    self.ada_ln2 = AdaLN(d_model)

    # Multi Linear Perceptron
    self.mlp = nn.Sequential(
        nn.Linear(d_model, d_model * 4),
        nn.SiLU(),
        nn.Linear(d_model * 4, d_model)
    )

    # Multi Head Attention
    self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)

    # Projection Layer to project time into a high-dimensional vector.
    self.time_proj = nn.Linear(d_model, d_model * 4)

  def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    t_params = self.time_proj(t).unsqueeze(1)
    scale1, shift1, scale2, shift2 = t_params.chunk(4, dim=-1)

    # Attention path
    x_norm1 = self.ada_ln1(x, scale1, shift1)
    attn_out, _ = self.attn(x_norm1, x_norm1, x_norm1)
    x = x + attn_out

    # MLP path
    x_norm2 = self.ada_ln2(x, scale2, shift2)
    mlp_out = self.mlp(x_norm2)
    x = x + mlp_out

    return x


class FlowDiT(nn.Module):
  """Full Architecture for generating Image."""
  def __init__(self, in_channels: int = 1, image_size: int = 28, patch_size: int = 7, d_model: int = 128, n_heads: int = 4, depth: int = 3):
    super().__init__()

    self.patch_size = patch_size
    self.seq_len = (image_size // patch_size) ** 2

    self.patch_embed = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)

    self.pos_embed = nn.Parameter(torch.randn(1, self.seq_len, d_model))
    self.time_mlp = nn.Sequential(
        SinusoidalTimeEmbedding(d_model),
        nn.Linear(d_model, d_model * 4),
        nn.SiLU(),
        nn.Linear(d_model * 4, d_model)
    )

    self.blocks = nn.ModuleList([DiTBlock(d_model, n_heads) for _ in range(depth)])

    self.head = nn.Linear(d_model, in_channels * patch_size * patch_size)

  def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    B, C, H, W = x.shape

    x = self.patch_embed(x).flatten(2).transpose(1, 2)

    x = x + self.pos_embed

    t_embed = self.time_mlp(t)

    for block in self.blocks:
      x = block(x, t_embed)

    x = self.head(x)

    P = self.patch_size
    H_out, W_out = H // P, W // P
    x = x.view(B, H_out, W_out, self.patch_embed.in_channels, P, P)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, C, H, W)

    return x