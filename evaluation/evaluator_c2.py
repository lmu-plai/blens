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

from evaluator import evaluation, makeDataFrame, findBLensInferenceFile
from metrics import loadVarCLRE, saveVarCLRE

def main(data_directory, VarCLR_cache, onlyA2=False):
	print("Ablation Study (C2)")
	A0 = ("COMBO (Table 6)", [('V11-PROJECTS-NO+UNK-DECODER+MULTI-LONG+', "BLens"), ('V11-PROJECTS-NO+UNK-DECODER+MULTI-LONG+-NO+COCA+V2', "BL-NP")])
	A1 = ("Input embeddings (Table 7)", [('V11-PROJECTS-NO+UNK-DECODER+MULTI-LONG+-DEXTER', "D"), ('V11-PROJECTS-NO+UNK-DECODER+MULTI-LONG+-PALMTREE', "P"), ('V11-PROJECTS-NO+UNK-DECODER+MULTI-LONG+-CLAP', "C"), ('V11-PROJECTS-NO+UNK-DECODER+MULTI-LONG+-PALMTREE+DEXTER', "P+D"),('V11-PROJECTS-NO+UNK-DECODER+MULTI-LONG+-Clap+Palmtree', "C+P"),('V11-PROJECTS-NO+UNK-DECODER+MULTI-LONG+-CLAP+DEXTER', "C+D"),('V11-PROJECTS-NO+UNK-DECODER+MULTI-LONG+', "C+P+D")])
	A2 = ("LORD (Table 8)", [('V11-PROJECTS-NO+UNK-DECODER+MULTI-LONG+', "LORD"),('V11-PROJECTS-NO+UNK-DECODER+SIMPLE-LONG+', "SIMPLE"),('V11-PROJECTS-NO+UNK-DECODER+MULTI-LONG+', "LORD-T0"),('V11-PROJECTS-NO+UNK-DECODER+SIMPLE-LONG+', "SIMPLE-T0"), ('V11-PROJECTS-NO+UNK-DECODER+MULTI-LONG+', "COMBO")])
	AL = [A0, A1, A2]
	if onlyA2:
		AL = [A2]

	for (title, experiments) in AL:
		print(title)

		experimentsS = []
		for (run, nameRun) in  experiments:
			experimentsS += [(nameRun, "xflBlensXProjectData", findBLensInferenceFile(data_directory, 'logs/blens/'+run, nameRun), "BLens")]

		df = makeDataFrame(data_directory, False, False, False, False, experimentsS)
		print(df)
		print()

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Evaluator of BLens Ablation (C2)")
	parser.add_argument('-data-dir', '--data-directory', dest="data_directory", default="../../data", help="The directory should contain data and logs used for BLens")

	# data_directory
	args = parser.parse_args()
	data_directory = args.data_directory

	# Load VarCLR embeddings
	VarCLR_cache = os.path.join(args.data_directory, 'embedding', 'varclrCache')
	loadVarCLRE(VarCLR_cache)

	pd.set_option("display.precision", 3)
	main(data_directory, VarCLR_cache, onlyA2=True)

	# Save new VarCLR embeddings
	saveVarCLRE(VarCLR_cache)
