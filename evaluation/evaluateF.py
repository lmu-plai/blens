"""
This file is part of the BLens binary function captioner.
Copyright (C) 2024-2025 by Tristan Benoit, Yunru Wang, Moritz Dannehl, and Johannes Kinder.
This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
"""



import os
import sys
import numpy as np
from tqdm import tqdm
from evaluation.metrics import micro_stats, metricBleu, metricRouge
import os
import evaluate

from builder import loadNLPData, loadData

def evaluateF(directoryXP, tokenizer, test, epoch=-1, specialCode=''):

	gtNames = {}

	for j in range(len(test)):
		(binPath, vaddr, _, _, tokens, _, _) = test[j]
		k = (binPath, vaddr)
		if not(k in gtNames):
			gtNames[k] = []
		nF = tokenizer.decode(tokens).replace("_", " ")
		if len(nF) > 0:
			gtNames[k] += [nF]

	pathLogs = os.path.join(directoryXP, f"LORD-eval-logs-{specialCode}.txt")

	original_stdout = sys.stdout
	original_stderr = sys.stderr

	fLogs = open(pathLogs, 'w')
	sys.stdout = fLogs
	sys.stderr = fLogs

	pathInfer = os.path.join(directoryXP,  f"LORD-inference-logs-{specialCode}.txt")

	calculableKnows = [ 'init', 'fini', 'csu_init', 'csu_fini', 'start' , 'libc_csu_init', 'libc_csu_fini', 'libc_start', 'deregister_tm_clones', 'register_tm_clones', 'rtld_init', 'main', 'do_global_dtors_aux', 'frame_dummy', 'frame_dummy_init_array_entry', 'do_global_dtors_aux_fini_array_entry', 'init_array_end', 'init_array_start', 'start_main', 'libc_start_main']

	target = None
	output = None

	pairs = []

	with open(pathInfer, "r") as f:
		for l in f.readlines():
			l = l.strip()
			if "target:" in l:
				if target != None:
					pairs += [[target, output]]
				target = l.split("target:")[1].replace(" ", "")

			if "output:" in l:
				output = l.split("output:")[1].replace(" ", "")

	if target != None:
		pairs += [[target, output]]

	predictions = []
	referencesBlue = []
	referencesRed = []

	pairsLabellized = []

	labels = {}
	idL = 0
	j = -1
	for [t, o] in tqdm(pairs):
		# The order is preserved by the builder and the builder remove no functions
		j += 1
		(binPath, vaddr, realFunctionName, functionName, tokens, bId, fId) = test[j]

		# If no token to predict we ignore this one in the evaluation
		if len(tokenizer.decode(tokens)) == 0:
			continue

		referencesBlue += [ gtNames[(binPath, vaddr)] ]
		referencesRed +=  [ gtNames[(binPath, vaddr)] ]

		if functionName in calculableKnows:
			predictions += [t.replace("_", " ")]
		else:
			predictions += [o.replace("_", " ")]

		tL = [0 for i in range(tokenizer.vocabulary)]
		oL = [0 for i in range(tokenizer.vocabulary)]

		for l in (o.split("_") + t.split("_")):
			if not(l in labels) and len(l) > 0:
				labels[l] = idL
				idL += 1

		for l in t.split("_"):
			if l in labels:
				tL[ labels[l] ] = 1

		if functionName in calculableKnows:
			oL = tL
		else:
			for l in o.split("_"):
				if l in labels:
					oL[ labels[l] ] = 1

		pairsLabellized += [ [tL, oL] ]

	print(len(labels), tokenizer.vocabulary)
	print(labels)

	bleuR = metricBleu.compute(predictions=predictions, references=referencesBlue, smooth=True)
	rougeR = metricRouge.compute(predictions=predictions, references=referencesRed)

	print(bleuR)
	print(rougeR)

	preds   = []
	targets = []

	for [tL, oL] in tqdm(pairsLabellized):
			tL = np.array(tL)
			oL = np.array(oL)

			if sum(tL) == 0:
				continue

			preds   += [oL]
			targets += [tL]

	targets = np.stack(targets)
	preds = np.stack(preds)
	print(targets.shape, preds.shape)
	f1, p, r = micro_stats(targets, preds)
	print(f"f1: {f1}, precision: {p}, recall: {r}")
	sys.stdout = original_stdout
	sys.stderr = original_stderr
	fLogs.close()
	return f1,p,r,bleuR,rougeR
