from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

def load_data():

	transform = transforms.Compose([
    	transforms.ToTensor(),
    	transforms.Normalize((0.5,), (0.5,))
  	])
	print(f"LOADING DATASET AND PREPARING...")
	dataset = datasets.MNIST(root="./", train=True, transform=transform, download=True)
	dataloader = DataLoader(dataset, shuffle=True, batch_size=256, drop_last=True)

	return dataloader