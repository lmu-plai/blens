"""
This file is part of the BLens binary function captioner.
Copyright (C) 2024-2025 by Tristan Benoit, Yunru Wang, Moritz Dannehl, and Johannes Kinder.
This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
"""



import os
import numpy as np
import torch
import sys
import re

from builder import restrictData, recursive_to_device
from COMBO.COMBO import COMBO

from decoding import decoding_bias

def inferenceCOMBO(directoryXP, params, tokenizer, testData, bias=0):
	torch.set_num_threads(16)
	original_stdout = sys.stdout
	original_stderr = sys.stderr
	pathLogs = os.path.join(directoryXP, f"COMBO-inference-logs.txt")
	fLogs = open(pathLogs, 'w')
	sys.stdout = fLogs
	sys.stderr = fLogs
	sizeTest = testData["caption_tokens"].size()[0]

	print(params, flush=True)

	combo = COMBO(tokenizer, params)
	if params["global"]["cuda"]:
		combo.cuda()
	combo.load_state_dict(torch.load(os.path.join(directoryXP, f"COMBO-Function-0")))
	combo.eval()

	batch_size =  params['global']['batch_size']
	decoder_type =  'simple'

	with torch.no_grad():
		for i in range(0, sizeTest, batch_size):
				batch = restrictData(testData, torch.from_numpy( np.arange(i, min(  i + batch_size, sizeTest))), cuda=params["global"]["cuda"])
				preds = decoding_bias(decoder_type, combo.decode(batch, decoder_type), bias)
				batch = recursive_to_device(batch, 'cpu')

				for j in range(len(preds)):
					cap = preds[j].tolist()
					cap = tokenizer.decode(cap)
					target = tokenizer.decode(batch["caption_tokens"][j].tolist())
					print('target: {}'.format(target))
					print('output: {}'.format(cap))
				del batch
				del preds

	sys.stdout = original_stdout
	sys.stderr = original_stderr
	fLogs.close()

