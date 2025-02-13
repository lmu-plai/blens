"""
This file is part of the BLens binary function captioner.
Copyright (C) 2024-2025 by Tristan Benoit, Yunru Wang, Moritz Dannehl, and Johannes Kinder.
This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
"""

import os
import argparse
import pandas as pd

from evaluator import evaluation, makeDataFrame, collectCSV
from metrics import loadVarCLRE, saveVarCLRE

def main(data_directory, VarCLR_cache):
	print("Main Results (C1)")

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

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Evaluator of BLens and related works (C1)")
	parser.add_argument('-data-dir', '--data-directory', dest="data_directory", default="../../data", help="The directory should contain data and logs used for BLens")

	# data_directory
	args = parser.parse_args()	
	data_directory = args.data_directory

	# Load VarCLR embeddings
	VarCLR_cache = os.path.join(args.data_directory, 'embedding', 'varclrCache')
	loadVarCLRE(VarCLR_cache)

	pd.set_option("display.precision", 3)
	main(data_directory, VarCLR_cache)

	# Save new VarCLR embeddings
	saveVarCLRE(VarCLR_cache)
