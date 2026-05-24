import torch
from loss import compute_flow_match_loss
from data_pipeline import load_data
from load_model import load_model

def train():

	device = "cuda" if torch.cuda.is_available() else "cpu"

	dataloader = load_data()

	model, optimizer = load_model()

	epochs = 25
	for epoch in range(epochs):
	
		model.train()

		total_loss = 0.0

		for batch_idx, (images, _) in enumerate(dataloader):

			images = images.to(device)

			optimizer.zero_grad()
			loss = compute_flow_match_loss(model, images)
			loss.backward()
			optimizer.step()

			total_loss += loss.item()

			if batch_idx % 100 == 0:
				print(f"Epoch: {epoch}| Batch idx: {batch_idx}| loss: {loss.item():.4f}")

	avg_loss = total_loss / len(dataloader)
	print(f"===Epoch: {epoch}| Average loss: {avg_loss:.4f} ===")

	return model