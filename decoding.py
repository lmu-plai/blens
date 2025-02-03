"""
This file is part of the BLens binary function captioner.
Copyright (C) 2024-2025 by Tristan Benoit, Yunru Wang, Moritz Dannehl, and Johannes Kinder.
This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
"""



import torch

def simple_decoding_bias(predictionsB, threshold):
	X = predictionsB['predictions'].clone()
	confidences, _ = torch.max(predictionsB['scores'], dim=-1)

	for b in range(len(X)):
		removing = False
		for t in range(1, len(X[b])): # first token of predictions are sos
			if removing == False and confidences[b, t-1] < threshold: # scores are after sos
				removing = True
			if removing:
				X[b,t] = 0
	return X.cpu()

def lord_decoding_bias(predictionsB, threshold):
	X = predictionsB['predictions'].clone()
	X[:, 0] = 0 # Put padding at the front
	X[X == 1] = 2 # EOS are encoded with 2 again instead of 1

	for b in range(len(predictionsB['scores'])):
		removing = False
		for (confidence,t,_) in predictionsB['scores'][b]:
			if removing == False and confidence < threshold:
				removing = True
			if removing:
				X[b, t] = 0
	return X.cpu()

def decoding_bias(decoder_type, predictionsB, threshold):
	if decoder_type == 'lord':
		return lord_decoding_bias(predictionsB, threshold)
	return simple_decoding_bias(predictionsB, threshold)
