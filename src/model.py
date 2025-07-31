import torch.nn as nn


class SKPredictor(nn.Module):
	def __init__(self, input_dim, output_dim, dropout_prob=0.2):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(input_dim, 512),
			nn.ReLU(),
			nn.Dropout(dropout_prob),
			nn.Linear(512, 256),
			nn.ReLU(),
			nn.Dropout(dropout_prob),
			nn.Linear(256, output_dim)
		)

	def forward(self, J_flat):
		logits = self.net(J_flat)
		return logits
