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

from evaluator_words import main as main_words
from evaluator_c1 import main as main_c1
from evaluator_c2 import main as main_c2

from metrics import loadVarCLRE, saveVarCLRE

def main(data_directory, VarCLR_cache):
	main_words(data_directory, VarCLR_cache)
	print("\n---\n")
	main_c1(data_directory, VarCLR_cache)
	print("\n---\n")
	main_c2(data_directory, VarCLR_cache)
	print("\n---\n")

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Complete evaluator of BLens")
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
