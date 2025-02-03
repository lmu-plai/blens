"""
This file is part of the BLens binary function captioner.
Copyright (C) 2024-2025 by Tristan Benoit, Yunru Wang, Moritz Dannehl, and Johannes Kinder.
This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
"""



import pandas as pd
import numpy as np
import pickle
import os

import argparse
from metrics import loadVarCLRE, stats, micro_stats, metricRouge, metricBleu, similarityVarCLR, saveVarCLRE

def loadNLPData(p):
	with open(p, "rb") as f:
		return pickle.load(f)  # [0,1,2] -> [binPath, vaddr, real_name, name, vaddr, bId, fId]

def translateResultsBLens(inferenceFile, nlpData, validation=False):
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

	if validation:
		nlpData = nlpData[1]
	else:
		nlpData = nlpData[2]

	data = []
	j = -1
	for [T, O] in pairs:
		j += 1
		(binPath, vaddr, realFunctionName, functionName, tokens, bId, fId) = nlpData[j]
		data += [ (binPath, vaddr, functionName, T, O, None) ]

	return data

def translateResultsXFL(pathLogs):
	data = []
	with open(pathLogs, 'r') as f:
		for line in f:
			line = line.strip() # /dbg_elf_bins/superiotool/usr/sbin/superiotool 30288 csu_fini csu_finalise ['csu', 'finalise'] => ['finalise', 'csu'] => ['csu', 'finalise']
			coordinate, entry, output  = line.split('=>')
			binPath, vaddr, functionName = coordinate.split(' ')[0:3]
			T = "_".join(eval(' '.join(line.split(' ')[4:]).split('=>')[0]))
			O = "_".join(eval(output))
			data += [ (binPath,int(vaddr), functionName, T, O, None) ]
	return data

def translateResultsSymLM(pathLogs, nlpData):
	coos = []
	with open(pathLogs.replace("_evaluation_input.txt", "-input.coo"), 'r') as f:
		for l in f.readlines():
			t = l.strip().split(" : ")
			coos += [(t[0], int(t[2]))]

	nlpData = nlpData[1]+nlpData[2]
	functionNames = {}
	for (binPath, vaddr, realFunctionName, functionName, tokens, bId, fId) in nlpData:
		functionNames[(binPath, vaddr)] = functionName

	data = []
	with open(pathLogs, 'r') as f:
		for i, line in enumerate(f):
			binPath, vaddr = coos[i]
			t = line.strip('\n').split(',')
			T = t[0].lower().strip().replace(" ", "_")
			O = t[1].lower().strip().replace(" ", "_")
			T = T.replace("<unk>_", "")
			T = T.replace("<pad>", "")
			probs = eval(t[2].replace(" ", ","))
			data += [ (binPath, vaddr, functionNames[(binPath, vaddr)], T, O, probs) ]
	return data

def translateResultsAsmDepictor(pathLogs):
	df = pd.read_csv(pathLogs, sep=",")
	df = df[["path","addr","name","tokenized name","asmdepictor"]]
	df = df.fillna('')
	data = []
	for (binPath,vaddr, functionName,  T, O) in list(df.itertuples(index=False, name=None)):
			data += [ (binPath, vaddr, functionName, T.replace(" ", "_"), O.replace(" ", "_"), None) ]
	return data

def translateResults(inferenceFile, method, nlpData, validation=False):
	if method == "BLens":
		return translateResultsBLens(inferenceFile, nlpData, validation=validation)
	if method == "XFL":
		return translateResultsXFL(inferenceFile)
	if method == "SymLM":
		return translateResultsSymLM(inferenceFile, nlpData)
	if method == "AsmDepictor":
		return translateResultsAsmDepictor(inferenceFile)
	return None

def createFunctionMask(p):
	d = {}
	with open(p, 'r') as f:
		for l in f.readlines():
			t = l.strip().split(" ")
			d[(t[0], int(t[1]))] = t[-1] == "True"
	return d

def readExtraDuplicates(p):
	d = {}
	with open(p, 'r') as f:
		for l in f.readlines():
			d[l.strip()] = True
	return d

def evaluation(data_directory, filterDuplicates, filterLabels, filterFree, filterExtra, nlpFold, inferenceFile, method, showLabels=False, showStrict=False, validation=False):
	calculableKnows = set([ 'init', 'fini', 'csu_init', 'csu_fini', 'start' , 'libc_csu_init', 'libc_csu_fini', 'libc_start', 'deregister_tm_clones', 'register_tm_clones', 'rtld_init', 'main', 'do_global_dtors_aux', 'frame_dummy', 'frame_dummy_init_array_entry', 'do_global_dtors_aux_fini_array_entry', 'init_array_end', 'init_array_start', 'start_main', 'libc_start_main'])
	forbiddenLabels = set(['dict', 'lk', 'emulator', 'vfs', 'ya', 'off', 'etc', 'initialize', 'pg', 'vmfs', 'diff', 'csu', 'cx', 'visit', 'xdr', 'av', 'pci', 'usal', 'mh', 'mutex', 'unpack', 'sc', 'rpl', 'ocaml', 'properties', 'notifier', 'fget', 'engine', '9', 'little', 'scsi', 'translate', 'ods', 'soap', 'mp', 'exception', 'clone', 'qapi', 'ikrt', 'curry'])
	extraDuplicates = readExtraDuplicates(os.path.join(data_directory, "strict_setting/forbidden_functions.txt"))
	
	if 'PROJECT' in nlpFold.upper() :
		fMask = createFunctionMask(os.path.join(data_directory, "strict_setting/projectHashFilterTest.csv"))
	else:
		fMask = createFunctionMask(os.path.join(data_directory, "strict_setting/binaryHashFilterTest.csv"))

	nlpData = loadNLPData(os.path.join(data_directory, nlpFold))
	resultsData = translateResults(os.path.join(data_directory, inferenceFile), method, nlpData, validation=validation)

	if method == "SymLM":
		threshold = optimizeSymLMThreshold(data_directory, filterDuplicates, filterLabels, filterFree, filterExtra, nlpFold, inferenceFile.replace("/test_", "/valid_"))
	else:
		threshold = 0

	# F1
	labels = {}
	occs = {}
	idL = 0
	pairsLabellized = []

	# RougeL, Bleu, VarCLR
	predictions = []
	references = []
	varCLRScores = []

	# Logs
	logs = []

	for (binPath, vaddr, functionName, t, o, probs) in resultsData:
		# Duplicates
		if filterDuplicates and fMask[(binPath, vaddr)]:
			continue

		# Extra duplicates based on ground truth
		if filterExtra and (t in extraDuplicates):
			continue

		# Filter forbidden words
		if filterLabels:
			t = "_".join([l for l in t.split("_") if not(l in forbiddenLabels)])
			# For SymLM we need to take care of words probabilities
			if not(probs is None):
				for i,l in enumerate(o.split("_")):
					if l in forbiddenLabels:
						probs[i] = 0
			else:
				o = "_".join([l for l in o.split("_") if not(l in forbiddenLabels)])

		# No words
		if t == "":
			continue

		# Free functions
		if functionName in calculableKnows:
			if filterFree:
				continue
			o = t


		# Apply SymLM threshold
		if not(probs is None) and not(functionName in calculableKnows): # Free functions do not take into account the condidence threshold
			ls = o.split("_")
			ls_kept = []
			for i, p in enumerate(probs):
				if float(p) > threshold:
					ls_kept.append(ls[i])
			o = "_".join(ls_kept)

		# F1
		tL = [0 for i in range(1024)]
		oL = [0 for i in range(1024)]

		for l in (o.split("_") + t.split("_")):
			if not(l in labels) and len(l) > 0:
				labels[l] = idL
				occs[l] = 0
				idL += 1

		for l in t.split("_"):
			tL[ labels[l] ] = 1

		if len(o) > 0:
			for l in o.split("_"):
				oL[ labels[l] ] = 1
				occs[l] += 1

		pairsLabellized += [ [tL, oL] ]

		# RougeL, Bleu, VarCLR
		references +=  [[t.replace("_", " ")]]
		predictions += [o.replace("_", " ")]
		varCLRScores += [similarityVarCLR(t,o)]

		# Logs
		logs += [(binPath, vaddr, functionName, t, o)]

	# F1
	preds   = []
	targets = []

	for [tL, oL] in pairsLabellized:
			tL = np.array(tL)
			oL = np.array(oL)

			if sum(tL) == 0:
				continue

			preds   += [oL]
			targets += [tL]

	targets = np.stack(targets)
	preds = np.stack(preds)
	f1, p, r = stats(targets, preds)
	micro_f1, micro_p, micro_r = micro_stats(targets, preds)

	# RougeL, Bleu, VarCLR
	rougeL = metricRouge.compute(predictions=predictions, references=references)['rougeL']
	bleu = metricBleu.compute(predictions=predictions, references=references, smooth=True)['bleu']
	varCLRScore = sum(varCLRScores)/len(varCLRScores)

	if showLabels:
		if showStrict: # Show how labels which occurs in the cross-project setting changed
			print("Occurences & F1 \\\\")
			for l in ['ocaml', 'get', 'string','free', 'type', 'initialise','fun', 'set', 'visit','soap', 'print', 'read','2', 'curry', 'path','name', 'usal', 'error','open', 'information']:
				if not(l in occs):
					print(l, "0 & X")
					continue
				print(l + " & " + str(occs[l]) + " & " + "{:.3f} \\\\".format(f1[labels[l]])	)
			print()
		else:
			print("Occurences &  Prec. & Recall & F1 \\\\")
			occsL = [(l, occs[l]) for l in occs]
			occsL = sorted(occsL, key=lambda loccs: -loccs[1])
			for (l, x) in occsL[:20]:
				print(l + " & " + str(x) + " & " + "{:.3f} & {:.3f} & {:.3f} \\\\".format(p[labels[l]],  r[labels[l]], f1[labels[l]])	)
			print()

	return micro_f1, micro_p, micro_r, rougeL, bleu, varCLRScore, logs

def optimizeSymLMThreshold(data_directory, filterDuplicates, filterLabels, filterFree, filterExtra, nlpFold, inferenceFile):
	calculableKnows = set([ 'init', 'fini', 'csu_init', 'csu_fini', 'start' , 'libc_csu_init', 'libc_csu_fini', 'libc_start', 'deregister_tm_clones', 'register_tm_clones', 'rtld_init', 'main', 'do_global_dtors_aux', 'frame_dummy', 'frame_dummy_init_array_entry', 'do_global_dtors_aux_fini_array_entry', 'init_array_end', 'init_array_start', 'start_main', 'libc_start_main'])
	forbiddenLabels = set(['dict', 'lk', 'emulator', 'vfs', 'ya', 'off', 'etc', 'initialize', 'pg', 'vmfs', 'diff', 'csu', 'cx', 'visit', 'xdr', 'av', 'pci', 'usal', 'mh', 'mutex', 'unpack', 'sc', 'rpl', 'ocaml', 'properties', 'notifier', 'fget', 'engine', '9', 'little', 'scsi', 'translate', 'ods', 'soap', 'mp', 'exception', 'clone', 'qapi', 'ikrt', 'curry'])
	extraDuplicates = readExtraDuplicates(os.path.join(data_directory, "strict_setting/forbidden_functions.txt"))

	if 'PROJECT' in nlpFold.upper() :
		fMask = createFunctionMask(os.path.join(data_directory, "strict_setting/projectHashFilterVal.csv"))
	else:
		fMask = createFunctionMask(os.path.join(data_directory, "strict_setting/binaryHashFilterVal.csv"))

	nlpData = loadNLPData(os.path.join(data_directory, nlpFold))
	resultsData = translateResults(os.path.join(data_directory, inferenceFile), 'SymLM', nlpData)

	labels = {}
	idL = 0
	pairs = []

	for (binPath, vaddr, functionName, t, o, probs) in resultsData:
		# Duplicates
		if filterDuplicates and fMask[(binPath, vaddr)]:
			continue

		# Extra duplicates based on ground truth
		if filterExtra and (t in extraDuplicates):
			continue

		# Filter forbidden words
		if filterLabels:
			t = "_".join([l for l in t.split("_") if not(l in forbiddenLabels)])
			for i,l in enumerate(o.split("_")):
				if l in forbiddenLabels:
					probs[i] = 0

		# No words
		if t == "":
			continue

		# Free functions
		if functionName in calculableKnows:
			if filterFree:
				continue
			o = t

		for l in (o.split("_") + t.split("_")):
			if not(l in labels) and len(l) > 0:
				labels[l] = idL
				idL += 1

		tL = [0 for i in range(1024)]
		for l in t.split("_"):
			tL[ labels[l] ] = 1

		pairs += [ [tL, o.split("_"), probs, functionName in calculableKnows] ]

	#print("Pairs", len(pairs))

	# Targets
	targets = []
	for [tL, _, _, _] in pairs:
		tL = np.array(tL)
		targets += [tL]
	targets = np.stack(targets)

	# Predictions

	bestF1 = 0
	bestThreshold = 0

	for t in range(100):
		threshold = t / 100.0

		preds   = []
		for [tL, o, probs, freeFunction] in pairs:
			if freeFunction == False: # Free functions do not take into account the condidence threshold
				lsk = []
				for i, p in enumerate(probs):
					if float(p) > threshold:
						lsk.append(o[i])
				o = lsk
			oL = np.zeros(1024)
			if len(o) > 0:
				pL = np.array([ labels[l] for l in o])
				np.put(oL, pL, 1)
			preds += [oL]
		preds = np.stack(preds)
		f1, _, _ = micro_stats(targets, preds)

		if f1 > bestF1:
			bestF1 = f1
			bestThreshold = threshold

	return bestThreshold

def makeDataFrame(data_directory, filterDuplicates, filterLabels, filterFree, filterExtra, experiments):
	df = {}
	for (method, nlpData, inferenceFile, translationMethod) in experiments:
		df[method] = {}
		micro_f1, micro_p, micro_r, rougeL, bleu, varCLRScore, _ = evaluation(data_directory, filterDuplicates, filterLabels, filterFree, filterExtra, nlpData, inferenceFile, translationMethod)
		df[method]["Precision"] = micro_p
		df[method]["Recall"] = micro_r
		df[method]["F1"] = micro_f1
		df[method]["RougeL"] = rougeL
		df[method]["Bleu"] = bleu
		df[method]["VarCLR"] = varCLRScore
	return pd.DataFrame(df).T

def collectCSV(data_directory, filterDuplicates, filterLabels, filterFree, filterExtra, experiments, output):
	defaultEntry = {}
	defaultEntry['groundtruth'] = ""
	for (method, _, _, _) in experiments:
		defaultEntry[method] = "<Not in the dataset>"

	D = {}
	for (method, nlpData, inferenceFile, translationMethod) in experiments:
		_, _, _, _, _, _, logs = evaluation(data_directory, filterDuplicates, filterLabels, filterFree, filterExtra, nlpData, inferenceFile, translationMethod)
		for (binPath, vaddr, functionName, t, o) in logs:
			coo = (binPath, vaddr, functionName)
			if not(coo in D):
				D[coo] = defaultEntry.copy()
			D[coo]['groundtruth'] = t
			D[coo][method] = o

	flattened_data = [{'binPath':coo[0], 'vaddr':coo[1], 'name':coo[2], **results} for coo, results in D.items()	]
	df = pd.DataFrame(flattened_data)
	df.to_csv(output, index=False)

def findBLensInferenceFile(data_directory, relativePath, nameRun):
	# Logs path
	run = os.path.join(data_directory, relativePath)

	# Special case for COMBO decoding
	if "COMBO" in nameRun:
		return os.path.join(relativePath, "COMBO-inference-logs.txt")

	# Look for the best epoch on the validation set
	bestValF1 = 0
	bEpoch = 0

	for e in range(200):
		if "T0" in nameRun:
			valLogs = os.path.join(run, f"LORD-eval-logs-val-nt-{e}.txt")
			if os.path.exists(valLogs) == False:
				continue
			with open(valLogs, 'r') as f:
				LF = [l.strip() for l in f.readlines()]
				LF = [l for l in LF if len(l) > 0]
				valF1 = float(LF[-1].split(" ")[1][:-1])
		else:
			valLogs = os.path.join(run, f"LORD-optimize-logs-val-{e}.txt")
			if os.path.exists(valLogs) == False:
				continue
			with open(valLogs, 'r') as f:
				LF = [l.strip() for l in f.readlines()]
				LF = [l for l in LF if len(l) > 0]
				valF1 = float(LF[-1].split(" ")[1])
		
		if valF1 > bestValF1:
			bestValF1 = valF1
			bEpoch = e
	
	# Return the correspondig inferrences on the test set (pre-computed by -inferBest), as a path from the data directory
	if "T0" in nameRun:
		return os.path.join(relativePath, f"LORD-inference-logs-test-nt-{bEpoch}.txt")
	return os.path.join(relativePath, f"LORD-inference-logs-test-{bEpoch}.txt")

def main():
	parser = argparse.ArgumentParser(description="Evaluator of BLens and related works")
	parser.add_argument('-data-dir', '--data-directory', dest="data_directory", default="../../data", help="The directory should contain data and logs used for BLens")

	# data_directory
	args = parser.parse_args()	
	data_directory = args.data_directory

	# Load VarCLR embeddings
	VarCLR_cache = os.path.join(args.data_directory, 'embedding', 'varclrCache')
	loadVarCLRE(VarCLR_cache)

	pd.set_option("display.precision", 3)

	print("Top 20 predicted words in the cross-binary setting. (Table 2)")
	evaluation(data_directory, False, False, True, False, 'xflBlensXBinaryData', 'logs/blens/V11-BINARIES-NO+UNK-DECODER+MULTI-LONG++/LORD-inference-logs-test-169.txt', 'BLens', showLabels=True)

	print("Top 20 words predicted in the cross-project setting. (Table 4)")
	evaluation(data_directory, False, False, True, False, 'xflBlensXProjectData', 'logs/blens/V11-PROJECTS-NO+UNK-DECODER+MULTI-LONG++/LORD-inference-logs-test-159.txt', 'BLens', showLabels=True)

	print("Top 20 words predicted in the cross-project setting with their occurences and F1 scores in the strict settings. (Table 4)")
	evaluation(data_directory, True, True, True, True, 'xflBlensXProjectData', 'logs/blens/V11-PROJECTS-NO+UNK-DECODER+MULTI-LONG++/LORD-inference-logs-test-159.txt', 'BLens', showLabels=True, showStrict=True)

	print()
	print("Main results")


	binaryExperiments =  [["BLens", 'xflBlensXBinaryData', 'logs/blens/V11-BINARIES-NO+UNK-DECODER+MULTI-LONG++/LORD-inference-logs-test-169.txt', 'BLens'],
				["XFL", 'xflBlensXBinaryData', 'logs/xfl/binaries.log', 'XFL'],
				["BL-S", 'symlmXBinaryData', 'logs/blens/V11-CCS-BINARIES-NO+UNK-DECODER+MULTI-LONG++/LORD-inference-logs-test-199.txt',  'BLens'],
				["SymLM", 'symlmXBinaryData', 'logs/symlm/binaries/test_evaluation_input.txt', 'SymLM'],
				["BL-A", 'asmdepictorXBinaryData', 'logs/blens/V11-ASIACCS-BINARIES-NO+UNK-DECODER+MULTI-LONG++/LORD-inference-logs-test-179.txt',  'BLens'],
				["AsmDepictor", 'asmdepictorXBinaryData', 'logs/asmdepictor/nlp_bin_asm.csv', 'AsmDepictor']]

	projectExperiments =  [["BLens", 'xflBlensXProjectData', 'logs/blens/V11-PROJECTS-NO+UNK-DECODER+MULTI-LONG++/LORD-inference-logs-test-159.txt', 'BLens'],
				["XFL", 'xflBlensXProjectData', 'logs/xfl/projects.log', 'XFL'],
				["BL-S", 'symlmXProjectData', 'logs/blens/V11-CCS-PROJECTS-NO+UNK-DECODER+MULTI-LONG++/LORD-inference-logs-test-189.txt',  'BLens'],
				["SymLM", 'symlmXProjectData', 'logs/symlm/projects/test_evaluation_input.txt', 'SymLM'],
				["BL-A", 'asmdepictorXProjectData', 'logs/blens/V11-ASIACCS-PROJECTS-NO+UNK-DECODER+MULTI-LONG++/LORD-inference-logs-test-159.txt',  'BLens'],
				["AsmDepictor", 'asmdepictorXProjectData', 'logs/asmdepictor/nlp_proj_asm.csv', 'AsmDepictor']]

	A = (False, False, False, False, [("cross-binary", '(Table 1)', binaryExperiments), ("cross-project", '(Table 3)', projectExperiments)])
	B = (True, False, True, False,  [("intermediate_strict_setting", '(Section 6.4)', projectExperiments)])
	C = (True, True, True, True,  [("strict_setting", '(Table 5)', projectExperiments)])

	for (filterDuplicates, FilterLabels, filterFree, filterExtra, xps) in [A,B,C]:
		for (stitle, table, experiments) in xps:
			print("Setting:", stitle, table)
			df = makeDataFrame(data_directory, filterDuplicates, FilterLabels, filterFree, filterExtra, experiments)
			print(df)
			collectCSV(data_directory, filterDuplicates, FilterLabels, filterFree, filterExtra, experiments, f"{stitle}.csv")
			print()
			print()

	print()
	print("Ablation Study")

	A0 = ("COMBO (Table 6)", [('V11-PROJECTS-NO+UNK-DECODER+MULTI-LONG+', "BLens"), ('V11-PROJECTS-NO+UNK-DECODER+MULTI-LONG+-NO+COCA+V2', "BL-OP")])
	A1 = ("Input embeddings (Table 7)", [('V11-PROJECTS-NO+UNK-DECODER+MULTI-LONG+-DEXTER', "D"), ('V11-PROJECTS-NO+UNK-DECODER+MULTI-LONG+-PALMTREE', "P"), ('V11-PROJECTS-NO+UNK-DECODER+MULTI-LONG+-CLAP', "C"), ('V11-PROJECTS-NO+UNK-DECODER+MULTI-LONG+-PALMTREE+DEXTER', "P+D"),('V11-PROJECTS-NO+UNK-DECODER+MULTI-LONG+-Clap+Palmtree', "C+P"),('V11-PROJECTS-NO+UNK-DECODER+MULTI-LONG+-CLAP+DEXTER', "C+D"),('V11-PROJECTS-NO+UNK-DECODER+MULTI-LONG+', "C+P+D")])
	A2 = ("LORD (Table 8)", [('V11-PROJECTS-NO+UNK-DECODER+MULTI-LONG+', "LORD"),('V11-PROJECTS-NO+UNK-DECODER+SIMPLE-LONG+', "SIMPLE"),('V11-PROJECTS-NO+UNK-DECODER+MULTI-LONG+', "LORD-T0"),('V11-PROJECTS-NO+UNK-DECODER+SIMPLE-LONG+', "SIMPLE-T0"), ('V11-PROJECTS-NO+UNK-DECODER+MULTI-LONG+', "COMBO")])

	for (title, experiments) in [A0, A1, A2]:
		print(title)

		experimentsS = []
		for (run, nameRun) in  experiments:
			experimentsS += [(nameRun, "xflBlensXProjectData", findBLensInferenceFile(data_directory, 'logs/blens/'+run, nameRun), "BLens")]

		df = makeDataFrame(data_directory, False, False, False, False, experimentsS)
		print(df)
		print()

	# Save new VarCLR embeddings
	saveVarCLRE(VarCLR_cache)

if __name__ == '__main__':
	main()
