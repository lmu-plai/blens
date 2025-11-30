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
from LORD.LORD import LORD
from decoding import decoding_bias

def inferenceLORD(directoryXP, params, tokenizer, testData, specialCode='', load_save=True, combo=None, lord=None, bias=0, epoch=-1):
	torch.set_num_threads(16)
	original_stdout = sys.stdout
	original_stderr = sys.stderr
	pathLogs = os.path.join(directoryXP, f"LORD-inference-logs-{specialCode}.txt")
	fLogs = open(pathLogs, 'w')
	sys.stdout = fLogs
	sys.stderr = fLogs
	sizeTest = testData["caption_tokens"].size()[0]

	print(params, flush=True)

	if load_save: 

		combo = COMBO(tokenizer, params)
		if params["global"]["cuda"]:
			combo.cuda()

		lord = LORD(tokenizer=tokenizer, params=params)
		if params["global"]["cuda"]:
			lord.cuda()

		if epoch > -1:
			checkpoint = torch.load(os.path.join(directoryXP, f"LORD-Function-0-epoch-{epoch}"))
			lord.load_state_dict(checkpoint['lord_model'])
			combo.load_state_dict(checkpoint['combo_model'])
			del checkpoint
		else:
			combo.load_state_dict(torch.load(os.path.join(directoryXP, f"COMBO-Function-F-0")))
			lord.load_state_dict(torch.load(os.path.join(directoryXP, f"LORD-Function-0")))

	combo.eval()
	lord.eval()

	batch_size =  params['global']['batch_size']
	decoder_type =  params['global']['decoder_type']

	with torch.no_grad():
		for i in range(0, sizeTest, batch_size):
				batch = restrictData(testData, torch.from_numpy( np.arange(i, min(  i + batch_size, sizeTest))), cuda=params["global"]["cuda"])
				batch["image_tokens"] = combo.image_embedding(batch)
				preds = decoding_bias(decoder_type, lord(batch), bias)
				batch = recursive_to_device(batch, 'cpu')

				for j in range(len(preds)):
					cap = preds[j].tolist()
					cap = tokenizer.decode(cap)
					target = tokenizer.decode(batch["caption_tokens"][j].tolist())
					print('target: {}'.format(target))
					print('output: {}'.format(cap))
				del batch
				del preds

	combo.train()
	lord.train()

	sys.stdout = original_stdout
	sys.stderr = original_stderr
	fLogs.close()

