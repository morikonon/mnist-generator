import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_flow_match_loss(model: nn.Module, x1: torch.Tensor) -> torch.Tensor:
  B = x1.shape[0]

  x0 = torch.randn_like(x1)

  t = torch.rand((B,), device=x1.device)
  t_expand = t.view(B, 1, 1, 1)

  xt = t_expand * x1 + (1 - t_expand) * x0

  target_velocity = x1 - x0

  pred_velocity = model(xt, t)

  loss = F.mse_loss(pred_velocity, target_velocity)

  return loss