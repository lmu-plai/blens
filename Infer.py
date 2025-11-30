"""
This file is part of the BLens binary function captioner.
Copyright (C) 2024-2025 by Tristan Benoit, Yunru Wang, Moritz Dannehl, and Johannes Kinder.
This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
"""

from argparse import ArgumentParser
import os
import pickle
import json

from builder import loadNLPData, loadData
from inferenceLORD import inferenceLORD

parser = ArgumentParser()

# Directories
parser.add_argument('-data-dir', '--data-directory', dest="data_directory", default="../data", help="The directory should contain data used for BLens")
parser.add_argument('-d', '--model-directory', dest="model_directory")

# Setting of the model
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--cross-project', action='store_true', help='Set the setting to cross-project.')
group.add_argument('--cross-binary', action='store_true', help='Set the setting to cross-binary.')

# Hyper parameters
parser.add_argument('-config', '--config', dest="config", default="main.json", help="The .json configuration file describing hyper parameters")

# Override parameters from the config file
parser.add_argument('--cpu', action='store_true', help='Disable CUDA')
parser.add_argument('-batch-size', '--batch-size', default=32, type=int)


args = parser.parse_args()
print(args)

ID_EXP = args.model_directory

# Load the dataset, the tokenizer and select DEXTER embeddings

nlpFold_filename = 'nlpDataTest'

if args.cross_project:
	tokenizer_name = "Tokenizer-Debin-1024-Projects"
else:
	tokenizer_name = 'Tokenizer-Debin-1024-Binaries'

nlpData = loadNLPData(os.path.join(args.data_directory, nlpFold_filename))
with open(os.path.join(args.data_directory, "tokenizer", tokenizer_name), "rb") as f:
	tokenizer = pickle.load(f)

with open(os.path.join('configs', args.config), "r") as f:
	params = json.load(f)
	


directoryXP = os.path.join(args.data_directory, 'xp', ID_EXP)
os.makedirs(directoryXP, exist_ok=True)

listOfFEmbeddings = []

if params["global"]["dexter"]:
	with open(os.path.join(args.data_directory, 'embedding', 'dexter_test'), "rb") as f:
		dexterEmbeddings  = pickle.load(f)
	listOfFEmbeddings += [("dexter", False, dexterEmbeddings)]

if params["global"]["clap"]:
	with open(os.path.join(args.data_directory, 'embedding',"clap_test"), "rb") as f:
		clapEmbeddings  = pickle.load(f)
	listOfFEmbeddings += [("clap", True, clapEmbeddings)]

if params["global"]["palmtree"]:
	with open(os.path.join(args.data_directory, 'embedding',"palmtree_test"), 'rb') as f:
		instructionSequences = pickle.load(f)
	listOfFEmbeddings += [("palmtree", True, instructionSequences)]

test = nlpData[2]
testData = loadData(test, tokenizer, params, listOfFEmbeddings, tokenizeGT=False)

params['global']['batch_size'] = args.batch_size

if args.cpu:
	params['global']['cuda'] = False

print('BLens Inference')
print(nlpFold_filename, tokenizer_name, len(test))
print(params)
print(ID_EXP)

bestValF1 = 0
bEpoch = 0
bThreshold = 0

for e in range(params['LORD']['epochs']):
	if (e+1) % params["global"]["interval"] == 0:

		with open(os.path.join(directoryXP, f"LORD-optimize-logs-val-{e}.txt"), 'r') as f:
			L = [l.strip() for l in f.readlines()]
			L = [l for l in L if len(l) > 0]

		threshold, valF1 = L[-1].split(" ")
		threshold = float(threshold)
		valF1 = float(valF1)

		if valF1 > bestValF1:
			bestValF1 = valF1
			bEpoch = e
			bThreshold = threshold

print('Found best model at epoch', bEpoch)
print('With threshold', bThreshold)

inferenceLORD(directoryXP, params, tokenizer, testData, specialCode=f'new-inferences', bias=bThreshold, epoch=bEpoch)

inferenceFile = os.path.join(directoryXP, f"LORD-inference-logs-new-inferences.txt")

target = None
output = None

pairs = []

with open(inferenceFile, "r") as f:
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

data = []
for j, [_, output] in enumerate(pairs):
	(binPath, vaddr, realFunctionName, functionName, tokens, bId, fId) = nlpData[2][j]
	print(binPath, vaddr, '->', output)

