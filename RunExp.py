"""
This file is part of the BLens binary function captioner.
Copyright (C) 2024-2025 by Tristan Benoit, Yunru Wang, Moritz Dannehl, and Johannes Kinder.
This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
"""



from argparse import ArgumentParser
import os
import secrets
import pickle
import json

from builder import loadNLPData, loadData

from pretrainCOMBO import pretrainCOMBO
from optimizeCOMBO import optimizeCOMBO
from inferenceCOMBO import inferenceCOMBO

from trainLORD import trainLORD
from inferenceLORD import inferenceLORD

from evaluation.evaluateF import evaluateF

parser = ArgumentParser()

# Basic test
parser.add_argument('--basic-test', dest="basic_test", action='store_true', help='The minimal test of BLens functionality.')
args, _ = parser.parse_known_args()

# Setting
if not args.basic_test:
	group = parser.add_mutually_exclusive_group(required=True)
	group.add_argument('--cross-project', action='store_true', help='Set the setting to cross-project.')
	group.add_argument('--cross-binary', action='store_true', help='Set the setting to cross-binary.')

# Directories
parser.add_argument('-data-dir', '--data-directory', dest="data_directory", default="../data", help="The directory should contain data and logs used for BLens")
parser.add_argument('-d', '--experiment-directory', dest="experiment_directory")

# Subdataset
group2 = parser.add_mutually_exclusive_group(required=False)
group2.add_argument('--symlm-subdataset', action='store_true', help='Employ the subdataset from SymLM pre-processing.')
group2.add_argument('--asmdepictor-subdataset', action='store_true', help='Employ the subdataset from AsmDepictor pre-processing.')

# Hyper parameters
parser.add_argument('-config', '--config', dest="config", default="main.json", help="The .json configuration file describing hyper parameters")

# Phases
parser.add_argument('-pretrain', '--pretrainCOMBO', dest="pretrainCOMBO", action='store_true')
parser.add_argument('-train', '--trainLORD', dest="trainLORD", action='store_true')
parser.add_argument('-inferBest', '--inferBest', dest="inferBest", action='store_true')

# Restart from an epoch
parser.add_argument('-loadEpoch', '--load-epoch',  type=int, dest="load_epoch", default=-1)

# Infer without the threshold (-T0)
parser.add_argument('-inferT0', '--inferT0', dest="inferT0", action='store_true')

# Reinfer without the threshold (-T0)
parser.add_argument('-reinferAllNoThreshold', '--reinferAllNoThreshold', dest="reinferAllNoThreshold", action='store_true')

# Inference with COMBO native decoder
parser.add_argument('-inferCOMBO', '--inferCOMBO', dest="inferCOMBO", action='store_true')

args = parser.parse_args()
print(args)

# Prepare a folder for the experiment
if args.experiment_directory == None:
	ID_EXP = secrets.token_hex(nbytes=8)
else:
	ID_EXP = args.experiment_directory

if args.basic_test:
    args.config = "basic-test.json"
    ID_EXP = 'basic_test'
    args.cross_project = True
    args.pretrainCOMBO = True
    args.trainLORD = True
    args.inferBest = True

# Load the dataset, the tokenizer and select DEXTER embeddings

if args.cross_project:
	if args.symlm_subdataset:
		nlpFold_filename = 'symlmXProjectData'
	elif args.asmdepictor_subdataset:
		nlpFold_filename = 'asmdepictorXProjectData'
	else:
		nlpFold_filename = 'xflBlensXProjectData'

	tokenizer_name = "Tokenizer-Debin-1024-Projects"
	dexterForSpit = 'dexterXProject'
else:
	if args.symlm_subdataset:
		nlpFold_filename = 'symlmXBinaryData'
	elif args.asmdepictor_subdataset:
		nlpFold_filename = 'asmdepictorXBinaryData'
	else:
		nlpFold_filename = 'xflBlensXBinaryData'

	tokenizer_name = 'Tokenizer-Debin-1024-Binaries'
	dexterForSpit = 'dexterXBinary'

nlpData = loadNLPData(os.path.join(args.data_directory, nlpFold_filename))

with open(os.path.join(args.data_directory, "tokenizer", tokenizer_name), "rb") as f:
	tokenizer = pickle.load(f)

with open(os.path.join('configs', args.config), "r") as f:
	params = json.load(f)

if params["global"]["no_combo"]:
	params["LORD"]["learning_rate_combo"] = params["LORD"]["learning_rate_lord"]
	params["COMBO"]["epochs"] = 0


directoryXP = os.path.join(args.data_directory, 'xp', ID_EXP)
os.makedirs(directoryXP, exist_ok=True)

if args.pretrainCOMBO:
	with open(os.path.join(directoryXP, "params"), "wb") as f:
		pickle.dump(params, f)
else:
	with open(os.path.join(directoryXP, "params"), "rb") as f:
		params = pickle.load(f)

listOfFEmbeddings = []

if params["global"]["dexter"]:
	with open(os.path.join(args.data_directory, 'embedding', dexterForSpit), "rb") as f:
		dexterEmbeddings  = pickle.load(f)
	listOfFEmbeddings += [("dexter", False, dexterEmbeddings)]

if params["global"]["clap"]:
	with open(os.path.join(args.data_directory, 'embedding',"clap"), "rb") as f:
		clapEmbeddings  = pickle.load(f)
	listOfFEmbeddings += [("clap", True, clapEmbeddings)]

if params["global"]["palmtree"]:
	with open(os.path.join(args.data_directory, 'embedding',"palmtree"), 'rb') as f:
		instructionSequences = pickle.load(f)
	listOfFEmbeddings += [("palmtree", True, instructionSequences)]

parameters = [tokenizer, params, listOfFEmbeddings]

train = nlpData[0]
val = nlpData[1]
test = nlpData[2]

if(args.basic_test):
	train = nlpData[0][23345:33345]
	val = nlpData[1][:1024]
	test = nlpData[2]

trainData = loadData(train, *parameters)
valData   = loadData(val, *parameters)
testData = loadData(test, *parameters)

print(nlpFold_filename, tokenizer_name, dexterForSpit)
print(params)
print(ID_EXP)
print(f"training set size: {trainData['caption_tokens'].size()[0]}, validation set size: {valData['caption_tokens'].size()[0]}, test set size: {testData['caption_tokens'].size()[0]}")

if args.pretrainCOMBO:
	pretrainCOMBO(directoryXP, params, tokenizer, trainData, valData, parameters)

if args.trainLORD:
	trainLORD(directoryXP, params, tokenizer, trainData, valData, testData, val, test, parameters, epochStart=args.load_epoch, noThreshold=args.inferT0)

if args.inferBest:
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

	inferenceLORD(directoryXP, params, tokenizer, testData, specialCode=f'test-{bEpoch}', bias=bThreshold, epoch=bEpoch)

if args.inferT0 or args.reinferAllNoThreshold:
	for e in range(params['LORD']['epochs']):
		if (e+1) % params["global"]["interval"] == 0:
			inferenceLORD(directoryXP, params, tokenizer, valData, specialCode=f'val-nt-{e}', bias=0, epoch=e)
			evaluateF(directoryXP, tokenizer,  val,  specialCode=f'val-nt-{e}', epoch=e)
			inferenceLORD(directoryXP, params, tokenizer, testData, specialCode=f'test-nt-{e}', bias=0, epoch=e)

if args.inferCOMBO:
	bias = optimizeCOMBO(directoryXP, params, tokenizer, valData, val)
	inferenceCOMBO(directoryXP, params, tokenizer, testData, bias=bias)
