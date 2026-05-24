import torch
import torch.nn as nn
import numpy as np

def generate(model: nn.Module, timesteps: int = 50):
    device = next(model.parameters()).device # Get device from model

    model.eval() # Set model to evaluation mode

    # Initial sample from standard normal distribution (noise)
    x_current = torch.randn((1, 1, 28, 28), device=device)

    # Perform Euler integration
    # Time steps for integration from t=0 to t=1
    dt = torch.tensor(1.0 / timesteps, device=device)
    t_values = torch.linspace(0, 1 - dt, timesteps, device=device)
    with torch.no_grad(): # Disable gradient calculations
        for i in range(len(t_values) - 1):
            t_start = t_values[i]
            t_end = t_values[i+1]

            # Model predicts the velocity v(x_t, t)
            # Unsqueeze t_start to match batch dimension (1,) for the model's time_mlp
            v_pred = model(x_current, t_start.unsqueeze(0))

            # Euler step: x_{t+dt} = x_t + v(x_t, t) * dt
            x_current = x_current + v_pred * dt

    # Post-process the generated image
    generated_image = x_current.cpu().squeeze().numpy() # Remove batch and channel dims, convert to numpy

    # Denormalize the image from [-1, 1] to [0, 1] (assuming original normalization was (0.5, 0.5))
    generated_image = (generated_image * 0.5) + 0.5

    # Clip to ensure pixel values are within [0, 1] bounds
    generated_image = np.clip(generated_image, 0, 1)
    
	return generated_image