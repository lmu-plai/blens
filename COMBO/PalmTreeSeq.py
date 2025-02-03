"""
This file is part of the BLens binary function captioner.
Copyright (C) 2024-2025 by Tristan Benoit, Yunru Wang, Moritz Dannehl, and Johannes Kinder.
This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
"""



import functools

import torch
import torch.nn as nn

class PalmTreeSeq(nn.Module):
	def __init__(
		self,
		size: int,
		visual_feature_size: int,
		palmtreeemb_size: int):

		super().__init__()
		self.size = size
		self.visual_feature_size = visual_feature_size
		self.palmtreeemb_size = palmtreeemb_size

		self.linear = nn.Linear(self.palmtreeemb_size, visual_feature_size)
		self.positions = nn.Embedding(size, visual_feature_size)

	def forward(self, X):
		batch_size, _, _ = X.size()
		X = self.linear(X)
		X = X + self.positions(self._create_position_indices(batch_size, X.device))
		return X

	@functools.lru_cache(maxsize=128)
	def _create_position_indices(self, batch_size, device):
		positions = torch.arange(self.size, dtype=torch.int32, device=device)
		positions = positions.unsqueeze(0).expand(batch_size, self.size)
		return positions
