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

from evaluator import evaluation
from metrics import loadVarCLRE, saveVarCLRE

def main(data_directory, VarCLR_cache):	
	print("Top 20 predicted words across settings\n\n")
	print("Top 20 predicted words in the cross-binary setting. (Table 2)")
	evaluation(data_directory, False, False, True, False, 'xflBlensXBinaryData', 'logs/blens/V11-BINARIES-NO+UNK-DECODER+MULTI-LONG++/LORD-inference-logs-test-169.txt', 'BLens', showLabels=True)

	print("Top 20 words predicted in the cross-project setting. (Table 4)")
	evaluation(data_directory, False, False, True, False, 'xflBlensXProjectData', 'logs/blens/V11-PROJECTS-NO+UNK-DECODER+MULTI-LONG++/LORD-inference-logs-test-159.txt', 'BLens', showLabels=True)

	print("Top 20 words predicted in the cross-project setting with their occurences and F1 scores in the strict settings. (Table 4)")
	evaluation(data_directory, True, True, True, True, 'xflBlensXProjectData', 'logs/blens/V11-PROJECTS-NO+UNK-DECODER+MULTI-LONG++/LORD-inference-logs-test-159.txt', 'BLens', showLabels=True, showStrict=True)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Evaluator of BLens word performance")
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
