"""
This file is part of the BLens binary function captioner.
Copyright (C) 2024-2025 by Tristan Benoit, Yunru Wang, Moritz Dannehl, and Johannes Kinder.
This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
"""



import argparse
import os
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle

oneColumnW = 241.020
twoColumnsW = 505.89
goldenRatio = (5**.5 - 1)/2

def update():
	plt.rcParams.update({
		#"text.usetex": True,
		"font.family": "DeJavu Serif",
		"font.serif": 'Times New Roman',
		"font.size": 16,
		"axes.titlesize": 16,
		"axes.labelsize": 16,
		"xtick.labelsize": 14,
		"ytick.labelsize": 14,
		"legend.fontsize": 16,
		"figure.titlesize": 16,
		#"text.latex.preamble": r'\usepackage{mathptmx}\usepackage[T1]{fontenc}\usepackage[utf8]{inputenc}\usepackage{pslatex}'
	})

# https://jwalton.info/Embed-Publication-Matplotlib-Latex/
def set_size(width, fraction, ratio):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * ratio

    fig_dim = (fig_width_in, fig_height_in)
    return fig_dim


def plotValidationF1(data_directory, experiments, output, width, fraction, ratio, framealpha=1):
	update()
	lines = ["-","--","-.",":"]
	linecycler = cycle(lines)
	fig, ax = plt.subplots(1, 1, figsize=set_size(width, fraction, ratio))
	mplotsL = []

	for u, (run, nameRun) in enumerate(experiments):
		run = os.path.join(data_directory, 'logs', 'blens', run)
		if "LONG++" in run:
			nE = 200
			period = 10
		else:
			nE = 80
			period = 4

		f1 = []
		E = []
		for e in range(nE):
			if (e+1) % period == 0:
				with open(os.path.join(run, f"LORD-optimize-logs-val-{e}.txt"), 'r') as f:
					LF = [l.strip() for l in f.readlines()]
					LF = [l for l in LF if len(l) > 0]
					threshold, micro_f1 = LF[-1].split(" ")
					threshold = float(threshold)
					micro_f1 = float(micro_f1)

				E += [e+1]
				f1 += [micro_f1]
		
		if nameRun=='Dexter':
			mplots = ax.plot(E, f1, next(linecycler), marker='x', markersize=3, linewidth=0.5, label=nameRun, color='#333A3F')
		elif u >= len(lines):
			mplots = ax.plot(E, f1, next(linecycler), marker='x', markersize=3, linewidth=0.5, label=nameRun)
		else:
			mplots = ax.plot(E, f1, next(linecycler), marker='o', markersize=3, linewidth=0.5, label=nameRun)

		mplotsL += [mplots[0]]

		annotations = []
		for i in range(len(E)):
			if f1[i] == max(f1):
				annotations += ["{:.3f}".format(f1[i])]
			else:
				annotations += ['']

		for i, txt in enumerate(annotations):
			xytext=(0,8)
			if nameRun=='C+D':
				xytext=(0,-15.5)
			ax.annotate(txt, (E[i], f1[i]), textcoords="offset points", xytext=xytext, ha='center', bbox=dict(facecolor='white', alpha=0.7, linewidth=1, edgecolor=plt.gca().lines[-1].get_color(),  pad=0.1, boxstyle='round'))

	#plt.autoscale()
	methods = [nameRun for (run, nameRun) in experiments]
	plt.xticks(np.arange(min(E), max(E)+1, period*2), rotation=65)
	fig.legend(mplotsL, methods, ncol=3, bbox_to_anchor=(0.9785, 0.46), facecolor='white', framealpha=framealpha, columnspacing=0.35)
	plt.xlabel("Epoch")
	plt.ylabel("Validation F1 score")
	plt.tight_layout()
	plt.grid()
	update()
	fig.savefig(output, dpi=1000, bbox_inches='tight')

parser = argparse.ArgumentParser(description="Evaluator of BLens and related works")
parser.add_argument('-data-dir', '--data-directory', dest="data_directory", default="../../data", help="The directory should contain data and logs used for BLens")

args = parser.parse_args()
data_directory = args.data_directory

print('Curve of validation F1 scores over the fine-tuning for the input embeddings ablation models. (Figure 5)')

XPs = [('V11-PROJECTS-NO+UNK-DECODER+MULTI-LONG+-DEXTER', "Dexter"), ('V11-PROJECTS-NO+UNK-DECODER+MULTI-LONG+-PALMTREE', "PalmTree"), ('V11-PROJECTS-NO+UNK-DECODER+MULTI-LONG+-CLAP', "Clap"), ('V11-PROJECTS-NO+UNK-DECODER+MULTI-LONG+-PALMTREE+DEXTER', "P+D"),('V11-PROJECTS-NO+UNK-DECODER+MULTI-LONG+-Clap+Palmtree', "C+P"),('V11-PROJECTS-NO+UNK-DECODER+MULTI-LONG+-CLAP+DEXTER', "C+D"),('V11-PROJECTS-NO+UNK-DECODER+MULTI-LONG+', "C+P+D")]
XPs.reverse()
plotValidationF1(data_directory, XPs, 'Figure_5.pdf', width=twoColumnsW, fraction=1., ratio=goldenRatio, framealpha=0.65)

print('Plot saved under', 'Figure_5.pdf')
