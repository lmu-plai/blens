"""
This file is part of the BLens binary function captioner.
Copyright (C) 2024-2025 by Tristan Benoit, Yunru Wang, Moritz Dannehl, and Johannes Kinder.
This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
"""



from functools import partial
from multiprocessing import Pool
import pickle
import os.path
import numpy as np

from NLP import NLP

class XFLTokenizerA:
	def __init__(self):
		self.idToLabel = {}
		self.labelToId = {}
		self.nlpMemory = {}

		self.pad_token_id = 0
		self.sos_token_id = 1
		self.eos_token_id = 2

		self.specialTokensThreshold =  -1
		self.vocabulary = 3

	@staticmethod
	def getCanonicalName(n):
		nlp = NLP()
		nP = n.replace('::', '_')
		r = nlp.tristan_canonical_name(nP)
		#print(n, r)
		return r

	def xflNLP(self, functionName):
		if not(functionName in self.nlpMemory):
			self.nlpMemory[functionName] = XFLTokenizerA.getCanonicalName(functionName)
		return self.nlpMemory[functionName]

	def loadInParallel(self, data):
		names = [data[j][3] for j in range(len(data)) if not(data[j][3] in self.nlpMemory)]

		names = list(set(names))

		with Pool(processes=16) as p:
			results = p.map(XFLTokenizerA.getCanonicalName, names)
		Y =  list(results)
		for j in range(len(names)):
			self.nlpMemory[names[j]] = Y[j]

	def inv_propensities(self, N, nLabels, A=0.5, B=0.425):
		"""
			calculate inverse propensity scores
			 P(y_t = 1 | y^*_t = 1)

			 N	  :: The size of the dataset
			 N_t	:: The number of data points annotated with label l
			 A, B   :: Hyperparameters
		"""
		L = len(self.labelToId)
		C = (np.log(N) - 1)*np.power((B + 1), A)

		inv_props = np.zeros((L, ), dtype=float)
		for t in self.labelToId:
			N_t = nLabels[t]
			exp_t = np.exp(-A * np.log(N_t + B))
			i_pt = 1.0 + (C * exp_t)
			inv_props[ self.labelToId[t] ] = i_pt
			print(str(t)+":"+str(N_t))
			print(str(t)+":"+str(i_pt))

		self.inv_props = inv_props

	def fit(self, allData, trainData, restrictTopK=-1, keepAllLabels=False, predefinedLabels={}):
		self.labelsToKeep = {}


		nLabelsTrain = {}
		for j in range(len(trainData)):
			for label in set(self.xflNLP(trainData[j][3]).split("_")):
				if not(label in nLabelsTrain):
					nLabelsTrain[label] = 1
					continue
				nLabelsTrain[label] += 1
		

		nLabels = {}
		for j in range(len(allData)):
			for label in set(self.xflNLP(allData[j][3]).split("_")):
				if not(label in nLabels):
					nLabels[label] = 1
					continue
				nLabels[label] += 1

		if restrictTopK > 0:
			# remove the labels that have less than 4 examples in the training set (XFL approach)
			tooFewInTraining = {}
			for n in nLabelsTrain:
				if nLabelsTrain[n] < 4:
					tooFewInTraining[n] = True
			
			topLabels = sorted( [(n, nLabels[n]) for n in nLabels if not(n in tooFewInTraining)], key = lambda x: (-x[1]))
			for (n, _)  in topLabels[:restrictTopK]:
				self.labelsToKeep[n] = True


		self.pad_token_id = 0
		self.labelToId["PAD"] = 0
		self.sos_token_id = 1
		self.labelToId["<s>"] = 1
		self.eos_token_id = 2
		self.labelToId["</s>"] = 2
		self.unk_token_id = 3
		self.labelToId["<unk>"] = 3

		nLabels["PAD"] = 0
		nLabels["<s>"] = len(allData)
		nLabels["</s>"] = len(allData)
		nLabels["<unk>"] = 0

		self.specialTokensThreshold = 4

		idL = 4
		for j in range(len(allData)):
			for label in self.xflNLP(allData[j][3]).split("_"):
				if len(predefinedLabels) > 0:
					if label in predefinedLabels and not(label in self.labelToId):
						self.labelToId[label] = idL
						idL += 1
					continue

				if keepAllLabels == False and not(label in self.labelsToKeep):
					continue

				if not(label in self.labelToId):
					self.labelToId[label] = idL
					idL += 1
					continue

		self.vocabulary = idL

		for label in self.labelToId:
			self.idToLabel[self.labelToId[label]] = label

		self.inv_propensities(len(allData), nLabels)

	def encode(self, nf, maxSize):
		tokens = []
		for label in self.xflNLP(nf).split("_"):
			if not(label in self.labelToId):
				tokens += [ self.unk_token_id ]
			else:
				tokens += [ self.labelToId[label] ]
		tokens = tokens[:maxSize]
		tokens = [self.sos_token_id] +  tokens + [self.eos_token_id]
		while len(tokens) < maxSize + 2:
			tokens += [self.pad_token_id]
		return 	tokens

	def transform(self, data, maxSize):
		self.loadInParallel(data)
		for j in range(len(data)):
			data[j][-3] = self.encode(data[j][3], maxSize)

	def decode(self, ids, skip_special_tokens=True):
		labels = []
		for idL in ids:
			if idL == self.eos_token_id:
				break
			if skip_special_tokens and idL < self.specialTokensThreshold:
				continue
			labels += [self.idToLabel[idL]]
		return "_".join(labels)


