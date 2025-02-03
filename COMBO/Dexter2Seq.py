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

class Dexter2Seq(nn.Module):
	def __init__(
		self,
		patches: int,
		dim: int,
		dimInter: int,
		visual_feature_size: int):
		super().__init__()
		self.patches = patches
		self.visual_feature_size = visual_feature_size
		self.dimInter = dimInter
		self.positions = nn.Embedding(patches,  visual_feature_size)
		self.linear = nn.Linear(dimInter, visual_feature_size)

	def forward(self, X):
		batch_size, _ = X.size()
		X = X.view(batch_size, self.patches, self.dimInter)
		X = self.linear(X)
		position_indices = self._create_position_indices(batch_size, X.device)
		position_embeddings = self.positions(position_indices)
		return X + position_embeddings

	@functools.lru_cache(maxsize=128)
	def _create_position_indices(self, batch_size, device):
		positions = torch.arange(self.patches, dtype=torch.int32, device=device)
		positions = positions.unsqueeze(0).expand(batch_size, self.patches)
		return positions
