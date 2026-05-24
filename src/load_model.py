from model import FlowDiT
import torch

def load_model():
	print(f"LOADING MODEL AND PREPARE OPTIMIZER...")
	device = "cuda" if torch.cuda.is_available() else "cpu"
	model = FlowDiT(in_channels=1, image_size=28, patch_size=4, d_model=256, n_heads=4, depth=6).to(device)
	optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

	print(f"MODEL PARAMETERS: {sum(p.numel() for p in model.parameters()) / 1000000}M ...")

	return model, optimizer