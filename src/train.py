import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset

from load_data import load_data
from model import SKPredictor


class SKDataset(Dataset):
	def __init__(self, data):
		self.data = data

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		J_flat, S = self.data[idx]
		J_flat = torch.tensor(J_flat, dtype=torch.float32)
		S_trimmed = torch.tensor(S[:-1], dtype=torch.float32)
		S_binary = (S_trimmed + 1) / 2
		return J_flat, S_binary


def train_model(model, train_loader, val_loader, device, epochs=50, lr=1e-3):
	criterion = nn.BCEWithLogitsLoss()
	optimizer = optim.Adam(model.parameters(), lr=lr)

	for epoch in range(epochs):
		model.train()
		total_loss = 0.0
		for inputs, targets in train_loader:
			inputs, targets = inputs.to(device), targets.to(device)
			optimizer.zero_grad()
			outputs = model(inputs)
			loss = criterion(outputs, targets)
			loss.backward()
			optimizer.step()
			total_loss += loss.item() * inputs.size(0)

		avg_train_loss = total_loss / len(train_loader.dataset)

		model.eval()
		val_loss = 0.0
		with torch.no_grad():
			for inputs, targets in val_loader:
				inputs, targets = inputs.to(device), targets.to(device)
				outputs = model(inputs)
				loss = criterion(outputs, targets)
				val_loss += loss.item() * inputs.size(0)

		avg_val_loss = val_loss / len(val_loader.dataset)
		print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}")


def main():
	N = 30
	num = 10000
	val_ratio = 0.2
	data: list[tuple[np.ndarray, np.ndarray]]
	val_indices: list[int]

	data, val_indices = load_data(N, num, val_ratio, regenerate=False)
	N = len(data[0][1])
	input_dim = N * (N - 1) // 2
	output_dim = N - 1

	val_set = Subset(SKDataset(data), val_indices)
	train_indices = list(set(range(len(data))) - set(val_indices))
	train_set = Subset(SKDataset(data), train_indices)

	train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
	val_loader = DataLoader(val_set, batch_size=64, shuffle=False)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = SKPredictor(input_dim=input_dim, output_dim=output_dim).to(device)

	train_model(model, train_loader, val_loader, device)


if __name__ == "__main__":
	main()
