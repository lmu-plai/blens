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
from evaluation.metrics import micro_stats

def optimizeLORD(directoryXP, params, tokenizer, valData, nlpVal, specialCode='', load_save=True, combo=None, lord=None, epoch=-1):
	torch.set_num_threads(16)
	original_stdout = sys.stdout
	original_stderr = sys.stderr
	pathLogs = os.path.join(directoryXP, f"LORD-optimize-logs-{specialCode}.txt")
	fLogs = open(pathLogs, 'w')
	sys.stdout = fLogs
	sys.stderr = fLogs
	sizeVal = valData["caption_tokens"].size()[0]

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

	labels = {}
	idL= 0
	for l in tokenizer.labelToId:
		if not(l in labels):
			labels[l] = idL
			idL += 1

	idToL = {}
	for l in labels:
		idToL[labels[l]] = l

	batch_size =  params['global']['batch_size']
	decoder_type =  params['global']['decoder_type']

	batchD = {}
	predictionsD = {}

	with torch.no_grad():
		for i in range(0, sizeVal, batch_size):
			batch = restrictData(valData, torch.from_numpy( np.arange(i, min(  i + batch_size, sizeVal))), cuda=params["global"]["cuda"])
			batch["image_tokens"] = combo.image_embedding(batch)
			predictionsD[i] = lord(batch)

			batch = recursive_to_device(batch, 'cpu')
			batch = [ tokenizer.decode(batch["caption_tokens"][j].tolist()) for j in range(len(batch["caption_tokens"]))]
			batch = [ [labels[l] for l in batch[j].split("_") if len(l) > 0] for j in range(len(batch))]
			batchD[i] = batch

	combo.train()
	lord.train()

	bestBias = 0
	bestF1 = 0

	calculableKnows = [ 'init', 'fini', 'csu_init', 'csu_fini', 'start' , 'libc_csu_init', 'libc_csu_fini', 'libc_start', 'deregister_tm_clones', 'register_tm_clones', 'rtld_init', 'main', 'do_global_dtors_aux', 'frame_dummy', 'frame_dummy_init_array_entry', 'do_global_dtors_aux_fini_array_entry', 'init_array_end', 'init_array_start', 'start_main', 'libc_start_main']

	for bias in np.linspace(0.0, 0.5, num=1000):
		totalTargets = []
		totalPreds = []
		indexNlp = 0

		for i in range(0, sizeVal, batch_size):
			batch = batchD[i]
			preds = decoding_bias(decoder_type, predictionsD[i], bias)

			preds =  [ preds[j].tolist() for j in range(len(preds)) ]
			preds =  [ tokenizer.decode(p) for p in preds]
			preds =  [ [labels[l] for l in preds[j].split("_") if len(l) > 0  ] for j in range(len(preds))]

			for j in range(len(preds)):
				(binPath, vaddr, realFunctionName, functionName, tokens, bId, fId) = nlpVal[indexNlp]
				indexNlp += 1
				if len(batch[j]) == 0:
					continue

				tL = np.zeros(tokenizer.vocabulary, dtype=int)
				oL = np.zeros(tokenizer.vocabulary, dtype=int)
				np.put(tL, batch[j], 1)
				np.put(oL, preds[j], 1)
				if functionName in calculableKnows:
					oL = tL
				totalTargets += [ tL ]
				totalPreds   += [ oL ]

		totalTargets = np.stack(totalTargets)
		totalPreds = np.stack(totalPreds)

		f1, p, r = micro_stats(totalTargets, totalPreds)
		print(bias, f1, p, r, flush=True)

		if f1 > bestF1:
			bestF1 = f1
			bestBias = bias

	print(bestBias, bestF1, flush=True)

	sys.stdout = original_stdout
	sys.stderr = original_stderr
	fLogs.close()
	return bestBias

