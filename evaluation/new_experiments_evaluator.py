"""
This file is part of the BLens binary function captioner.
Copyright (C) 2024-2025 by Tristan Benoit, Yunru Wang, Moritz Dannehl, and Johannes Kinder.
This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
"""

import os
import pandas as pd
import argparse

from metrics import loadVarCLRE, saveVarCLRE
from evaluator import findBLensInferenceFile, makeDataFrame, collectCSV

def main():
	pd.set_option("display.precision", 3)

	parser = argparse.ArgumentParser(description="Simple evaluator for BLens new experiments")
	parser.add_argument('--name', type=str, default='test', help='Name of the XP to print (default: "test").')

	# Directories
	parser.add_argument('-data-dir', '--data-directory', dest="data_directory", default="../../data", help="The directory should contain data and logs used for BLens")
	parser.add_argument('-d', '--experiment-directory', dest="experiment_directory")

	# Setting
	group = parser.add_mutually_exclusive_group(required=True)
	group.add_argument('--cross-project', action='store_true', help='Set the setting to cross-project.')
	group.add_argument('--cross-binary', action='store_true', help='Set the setting to cross-binary.')

	# Subdataset
	group2 = parser.add_mutually_exclusive_group(required=False)
	group2.add_argument('--symlm-subdataset', action='store_true', help='Employ the subdataset from SymLM pre-processing.')
	group2.add_argument('--asmdepictor-subdataset', action='store_true', help='Employ the subdataset from AsmDepictor pre-processing.')

	# Ablation parameters
	parser.add_argument('--evaluateNoThreshold', action='store_true', default=False, help='Evaluate without a threshold (default: False).')
	parser.add_argument('--evaluateComboDecoder', action='store_true', default=False, help='Evaluate combo decoder (default: False).')

	# Parse the arguments
	args = parser.parse_args()
	nameXP = args.name
	data_directory = args.data_directory
	xp_directory = os.path.join('xp', args.experiment_directory)

	# Load VarCLR embeddings
	VarCLR_cache = os.path.join(data_directory, 'embedding', 'varclrCache')
	loadVarCLRE(VarCLR_cache)

	# Load the dataset
	if args.cross_project:
		if args.symlm_subdataset:
			nlpData = 'symlmXProjectData'
		elif args.asmdepictor_subdataset:
			nlpData = 'asmdepictorXProjectData'
		else:
			nlpData = 'xflBlensXProjectData'
	else:
		if args.symlm_subdataset:
			nlpData = 'symlmXBinaryData'
		elif args.asmdepictor_subdataset:
			nlpData = 'asmdepictorXBinaryData'
		else:
			nlpData = 'xflBlensXBinaryData'

	print('BLens', 'cross-binary setting' if args.cross_binary else 'cross-project setting')
	print(xp_directory, nameXP)
	print()
	print()

	# Ablations
	A = []
	if args.evaluateNoThreshold:
		A += [("without a threshold", [nameXP+"-T0"])]
	if args.evaluateComboDecoder:
		A += [("COMBO decoder", [nameXP+"-COMBO"])]

	if len(A) > 0:
		print("Ablations")
		for (title, experiments) in  A:
			print(title)
			experimentsS = []
			for nameRun in  experiments:
				experimentsS += [(nameRun, nlpData, findBLensInferenceFile(data_directory, xp_directory, nameRun), "BLens")]
			df = makeDataFrame(data_directory, False, False, False, False, experimentsS)
			print(df)
			print()
		return

	# Main results
	print("Main results")
	XPs = [[nameXP, nlpData, findBLensInferenceFile(data_directory, xp_directory, nameXP), 'BLens']]

	A = (False, False, False, False, "cross-project")
	B = (True, False, True, False,  "intermediate strict")
	C = (True, True, True, True,  "strict")

	if args.cross_project:
		settings = [A,B,C]
	else:
		settings = [(False, False, False, False, "cross-binary")]


	for (filterDuplicates, FilterLabels, filterFree, filterExtra, stitle) in settings:
		print("Setting:", stitle)
		df = makeDataFrame(data_directory, filterDuplicates, FilterLabels, filterFree, filterExtra, XPs)
		print(df)
		print()


	# Save new VarCLR embeddings
	saveVarCLRE(VarCLR_cache)

if __name__ == '__main__':
	main()
