"""
This file is part of the BLens binary function captioner.
The 'ml_precision', 'ml_recall', 'ml_f1', 'ml_f1', 'ml_tp', 'ml_fn', 'ml_fp', 'stats' and 'micro_stats' functions are components from the official XFL implementation by James Patrick-Evans, Moritz Dannehl, and Johannes Kinder, licensed under the GPL-v3 license.

Copyright (c) 2017-2024 James Patrick-Evans, Moritz Dannehl, and Johannes Kinder.
Portions Copyright (c) 2024-2025 Tristan Benoit, Yunru Wang, Moritz Dannehl, and Johannes Kinder.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
"""



import numpy as np
import evaluate
import torch.nn.functional as F
from varclr.models.model import Encoder
import os
import pickle
from pathlib import Path

np.seterr(invalid='ignore')


def ml_precision(true_Y, pred_Y, MODE='MICRO') -> np.ndarray:
	"""
		Calculate ML Precision from true label and predicted label matrix
	"""
	assert(MODE in ('MICRO', 'MACRO'))
	tp  = ml_tp(true_Y, pred_Y, MODE)
	fp  = ml_fp(true_Y, pred_Y, MODE)
	return tp / (tp + fp)

def ml_recall(true_Y, pred_Y, MODE='MICRO') -> np.ndarray:
	"""
		Calculate ML Recall from ture label and predicted label matrix
	"""
	assert(MODE in ('MICRO', 'MACRO'))
	tp  = ml_tp(true_Y, pred_Y, MODE)
	fn  = ml_fn(true_Y, pred_Y, MODE)
	return tp / (tp + fn)


def ml_f1(p, r):
	"""
		Calculate ML F1 from ML Precision and ML Recall
	"""
	f1 = (2 * p * r) / ( p + r)
	return f1

def ml_tp(true_Y, pred_Y, MODE:str) -> np.ndarray:
	"""
		Expects 0's and 1's in both true and predicted label set
	"""
	assert(true_Y.shape == pred_Y.shape)
	r, c    = true_Y.shape
	##tp vector for each label
	tp      = np.zeros((c, ))
	#loop over all nonzero labels in true prediction
	rz, cz  = true_Y.nonzero()

	for i in range(len(rz)):
		ri  = rz[i]
		ci  = cz[i]
		if pred_Y[ri, ci] == true_Y[ri, ci]:
			tp[ci]  += 1

	#calculates tp for all classes, rather than vector of tp's
	if MODE=='MICRO':
		return np.sum(tp)
	return tp

def ml_fn(true_Y, pred_Y, MODE:str) -> np.ndarray:
	"""
		MultiLabel Flase Negative.
			Labels missed in prediction but exist in true set
	"""
	assert(true_Y.shape == pred_Y.shape)
	r, c    = true_Y.shape
	##tp vector for each label
	fn      = np.zeros((c, ))
	#loop over all nonzero labels in true prediction
	rz, cz  = true_Y.nonzero()

	for i in range(len(rz)):
		ri  = rz[i]
		ci  = cz[i]
		assert(true_Y[ri, ci] == 1)
		if pred_Y[ri, ci] != true_Y[ri, ci]:
			fn[ci]  += 1

	##calculates fn for all classes
	if MODE=='MICRO':
		return np.sum(fn)
	return fn

def ml_fp(true_Y, pred_Y, MODE:str) -> np.ndarray:
	"""
		MultiLabel Flase Positive.
			Labels predicted but don't exist in true set
	"""
	assert(true_Y.shape == pred_Y.shape)
	r, c    = true_Y.shape
	##tp vector for each label
	fp      = np.zeros((c, ))
	#loop over all nonzero labels in true prediction
	rz, cz  = pred_Y.nonzero()

	for i in range(len(rz)):
		ri  = rz[i]
		ci  = cz[i]
		if pred_Y[ri, ci] != true_Y[ri, ci]:
			fp[ci]  += 1
	##calculates fp for all classes
	if MODE=='MICRO':
		return np.sum(fp)
	return fp

def stats(targets, preds):
    p = ml_precision(targets, preds, MODE="MACRO")
    r = ml_recall(targets, preds, MODE="MACRO")
    f1 = ml_f1(p, r)
    return (f1,p,r)

def micro_stats(targets, preds):
    p = ml_precision(targets, preds)
    r = ml_recall(targets, preds)
    f1 = ml_f1(p, r)
    return (f1,p,r)


metricRouge = evaluate.load('rouge')
metricBleu = evaluate.load("bleu")
varCLR = Encoder.from_pretrained("varclr-codebert")

varCLRE = {}

def loadVarCLRE(path):
	print("Retrieving VarCLR embeddings:", path)
	global varCLRE
	if os.path.isfile(path):
		with open(path, "rb") as f:
			varCLRE = pickle.load(f)		
	print("Cached VarCRL embeddings:", len(varCLRE))
	print()
	print()

def projectVarCLR(n):
	global varCLRE, varCLR
	if not(n in varCLRE):
		varCLRE[n] = varCLR.encode(n)
	return varCLRE[n]

def similarityVarCLR(x,y):
	return F.cosine_similarity(projectVarCLR(x), projectVarCLR(y)).item()

def saveVarCLRE(path):
	global varCLRE
	print("Saving VarCLR embeddings:", path)
	with open(path, "wb") as f:
		pickle.dump(varCLRE, f)
