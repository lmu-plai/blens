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
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR
from tqdm import tqdm
from random import seed, shuffle
import sys
import os

from builder import restrictData, loadNLPData, loadData

from COMBO.COMBO import COMBO
from LORD.LORD import LORD
from optimizeLORD import optimizeLORD

def trainLORD(directoryXP, params, tokenizer, trainData, valData, testData, val, test, parameters, epochStart=-1):
	torch.set_num_threads(params['global']['threads'])

	pathCOMBOBase = os.path.join(directoryXP, f"COMBO-Function-0")
	pathCOMBO = os.path.join(directoryXP, f"COMBO-Function-F-0")
	pathLORD = os.path.join(directoryXP, f"LORD-Function-0")
	pathLogs = os.path.join(directoryXP, f"LORD-logs.txt")

	original_stdout = sys.stdout
	original_stderr = sys.stderr

	fLogs = open(pathLogs, 'a+')
	sys.stdout = fLogs
	sys.stderr = fLogs

	print(params, flush=True)

	sizeTrain = trainData["caption_tokens"].size()[0]
	sizeVal = valData["caption_tokens"].size()[0]

	combo = COMBO(tokenizer, params)
	if params["global"]["cuda"]:
		combo.cuda()
	combo_weights = torch.load(pathCOMBOBase)
	combo.load_state_dict(combo_weights)

	lord = LORD(tokenizer, combo_weights, params)

	if params["global"]["cuda"]:
		lord.cuda()

	lord.train()


	print("Total Ensemble Encoder parameters", combo.ensembleEncoderParameters(), flush=True)
	print("Total Function Encoder parameters", combo.functionEncoderParameters(), flush=True)
	print("Total LORD parameters", sum(p.numel() for p in lord.parameters() if p.requires_grad), flush=True)

	batch_size =  params['global']['batch_size']
	learning_rate_combo = params['LORD']['learning_rate_combo']
	learning_rate_lord = params['LORD']['learning_rate_lord']
	epochs = params['LORD']['epochs']
	max_grad_norm = params['LORD']['max_grad_norm']
	weight_decay = params['LORD']['weight_decay']
	warmup = int(params['LORD']['warmup_percent'] * (epochs*sizeTrain) / ( batch_size *  100)) + 1

	optimizerCOMBO = torch.optim.AdamW(combo.parameters(), lr = learning_rate_combo, weight_decay=weight_decay)
	warmupSchedulerCOMBO = LambdaLR(optimizerCOMBO, lr_lambda=[lambda step: step / warmup])
	cosineSchedulerCOMBO = CosineAnnealingLR(optimizerCOMBO, T_max=int((epochs*sizeTrain) /  (batch_size*params['global']['cosine_annealing_steps']) ), eta_min=0)
	schedulerCOMBO = SequentialLR(optimizerCOMBO, schedulers=[warmupSchedulerCOMBO, cosineSchedulerCOMBO], milestones=[warmup])

	optimizerLORD = torch.optim.AdamW(lord.parameters(), lr = learning_rate_lord, weight_decay=weight_decay)
	warmupSchedulerLORD = LambdaLR(optimizerLORD, lr_lambda=[lambda step: step/ warmup])
	cosineSchedulerLORD = CosineAnnealingLR(optimizerLORD,  T_max=int((epochs*sizeTrain) /  (batch_size*params['global']['cosine_annealing_steps']) ), eta_min=0)
	schedulerLORD = SequentialLR(optimizerLORD, schedulers=[warmupSchedulerLORD, cosineSchedulerLORD], milestones=[warmup])


	if epochStart > -1:
		checkpoint = torch.load(pathLORD+f"-epoch-{epochStart}")
		schedulerLORD.load_state_dict(checkpoint['lord_scheduler'])
		lord.load_state_dict(checkpoint['lord_model'])
		optimizerLORD.load_state_dict(checkpoint['lord_optimizer'])
		schedulerCOMBO.load_state_dict(checkpoint['combo_scheduler'])
		combo.load_state_dict(checkpoint['combo_model'])
		optimizerCOMBO.load_state_dict(checkpoint['combo_optimizer'])
		del checkpoint

	minLoss = 100000

	for epoch in range(epochs):

		if epoch <= epochStart:
			continue

		combo.train()

		lossS = 0
		permutation = torch.randperm(sizeTrain)
		for i in tqdm(range(0, sizeTrain, batch_size)):
			optimizerLORD.zero_grad()
			optimizerCOMBO.zero_grad()

			indices = permutation[i:i+batch_size]
			batch = restrictData(trainData, indices, cuda=params["global"]["cuda"], combo=True)

			image_seq = combo.image_embedding(batch)

			batch["image_tokens"] = image_seq
			loss = lord(batch)
			loss.backward()
			lossS += loss.cpu().item()
			torch.nn.utils.clip_grad_norm_(combo.parameters(), max_norm = max_grad_norm)
			torch.nn.utils.clip_grad_norm_(lord.parameters(), max_norm = max_grad_norm)
			optimizerLORD.step()
			optimizerCOMBO.step()
			schedulerLORD.step()
			schedulerCOMBO.step()
			del batch, loss, image_seq


		lossS /= (sizeTrain /  batch_size )
		print("Training", epoch, lossS, flush=True)
		combo.eval()
		with torch.no_grad():
			lossS = 0

			permutation = torch.randperm(sizeVal)
			for i in tqdm(range(0, sizeVal, batch_size)):
				indices = permutation[i:i+batch_size]
				batch = restrictData(valData, indices, cuda=params["global"]["cuda"], combo=True)

				image_seq = combo.image_embedding(batch)
				batch["image_tokens"] = image_seq
				lossS += lord(batch).cpu().item()
				del batch, image_seq
			
			lossS /= (sizeVal /  batch_size )
			print("Validation", epoch, lossS, flush=True)
			if lossS < minLoss:
				print("New minima", minLoss, "->" , lossS, flush=True)
				minLoss = lossS

			if((epoch+1)%params["global"]["interval"]==0):
				optimizeLORD(directoryXP, params, tokenizer, valData, val, specialCode=f"val-{epoch}",load_save=False, combo=combo, lord=lord)
				torch.save({'lord_scheduler':schedulerLORD.state_dict(), 'lord_model':lord.state_dict(), 'lord_optimizer':optimizerLORD.state_dict(), 'combo_scheduler':schedulerCOMBO.state_dict(), 'combo_model':combo.state_dict(), 'combo_optimizer':optimizerCOMBO.state_dict()}, pathLORD+f"-epoch-{epoch}")


	torch.save(lord.state_dict(), pathLORD)
	torch.save(combo.state_dict(), pathCOMBO)

	sys.stdout = original_stdout
	sys.stderr = original_stderr
	fLogs.close()
