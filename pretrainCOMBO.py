"""
This file is part of the BLens binary function captioner.
Copyright (C) 2024-2025 by Tristan Benoit, Yunru Wang, Moritz Dannehl, and Johannes Kinder.
This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
"""



import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR

from tqdm import tqdm
from random import seed, shuffle
import sys
import os
from builder import restrictData
from COMBO.COMBO import COMBO
from builder import loadNLPData, loadData

def pretrainCOMBO(directoryXP, params, tokenizer, trainData, valData, parameters):
	torch.set_num_threads(params['global']['threads'])

	pathCOMBO = os.path.join(directoryXP, f"COMBO-Function-0")
	pathLogs = os.path.join(directoryXP, f"COMBO-logs.txt")

	original_stdout = sys.stdout
	original_stderr = sys.stderr

	fLogs = open(pathLogs, 'w')
	sys.stdout = fLogs
	sys.stderr = fLogs

	print(params, flush=True)

	sizeTrain = trainData["caption_tokens"].size()[0]
	sizeVal = valData["caption_tokens"].size()[0]

	model = COMBO(tokenizer, params)
	if params["global"]["cuda"]:
		model.cuda()

	print("Total COMBO parameters", sum(p.numel() for p in model.parameters() if p.requires_grad), flush=True)

	batch_size =  params['global']['batch_size']
	learning_rate = params['COMBO']['learning_rate']
	epochs = params['COMBO']['epochs']
	max_grad_norm = params['COMBO']['max_grad_norm']
	weight_decay = params['COMBO']['weight_decay']
	warmup = int(params['COMBO']['warmup_percent'] * (epochs*sizeTrain) / ( batch_size *  100)) + 1

	optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate, weight_decay=weight_decay)
	warmupScheduler = LambdaLR(optimizer, lr_lambda=[lambda step: step / warmup])
	cosineScheduler = CosineAnnealingLR(optimizer, T_max=int((epochs*sizeTrain) /  (batch_size* params['global']['cosine_annealing_steps'] ) ), eta_min=0)
	scheduler = SequentialLR(optimizer, schedulers=[warmupScheduler, cosineScheduler], milestones=[warmup])

	minLoss = 100000

	for epoch in range(epochs):
		print(f"Epoch:{epoch}")
		model.train()
		lossS = 0
		lossSCa = 0
		lossSCo = 0
		permutation = torch.randperm(sizeTrain)
		for i in tqdm(range(0, sizeTrain, batch_size)):
			optimizer.zero_grad()
			indices = permutation[i:i+batch_size]
			batch = restrictData(trainData, indices, cuda=params["global"]["cuda"], combo=True)
			lossCa, lossCo = model(batch)
			(lossCa + lossCo).backward()
			lossSCa += lossCa.cpu().item()
			lossSCo += lossCo.cpu().item()
			lossS += lossCa.cpu().item() + lossCo.cpu().item()
			torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = max_grad_norm)
			optimizer.step()
			scheduler.step()
			del batch, lossCa, lossCo


		lossSCa /= (sizeTrain / batch_size )
		lossSCo /= (sizeTrain / batch_size )
		lossS /= (sizeTrain / batch_size )
		print("combo-training-loss:", lossS, "combo-tr-ca:", lossSCa, "combo-tr-co:", lossSCo)

		model.eval()
		with torch.no_grad():
			lossS = 0
			lossSCa = 0
			lossSCo = 0

			permutation = torch.randperm(sizeVal)
			for i in tqdm(range(0, sizeVal, batch_size)):
				indices = permutation[i:i+batch_size]
				batch = restrictData(valData, indices, cuda=params["global"]["cuda"], combo=True)
				lossCa, lossCo = model(batch)
				lossSCa += lossCa.cpu().item()
				lossSCo += lossCo.cpu().item()
				lossS += lossCa.cpu().item() + lossCo.cpu().item()
				del batch, lossCa, lossCo

			lossSCa /= (sizeVal /  batch_size )
			lossSCo /= (sizeVal /  batch_size )
			lossS /= (sizeVal /  batch_size )
			print("combo-validation-loss:", lossS, "combo-va-ca:", lossSCa, "combo-va-co:", lossSCo)
			if lossS < minLoss:
				print("New minima", minLoss, "->" , lossS, flush=True)
				minLoss = lossS

	torch.save(model.state_dict(), pathCOMBO)

	sys.stdout = original_stdout
	sys.stderr = original_stderr
	fLogs.close()


