# BLens: Contrastive Captioning of Binary Functions using Ensemble Embedding
**Authors**: Tristan Benoit\*, Yunru Wang\*, Moritz Dannehl, Johannes Kinder

This is the artifact accompanying the paper _BLens: Contrastive Captioning of Binary Functions using Ensemble Embedding_, appearing in the 34th USENIX Security Symposium, August 13-15, in Seattle, WA, USA (USENIX Security 2025).

The artifact contains:

- [data]`data/`: Train, validation and test splits used for both cross-binary and cross-project settings
- [data]`data/embedding/`: Pre-computed DEXTER, CLAP, and PalmTree embeddings
- [data]`data/logs/`: Pre-computed logs for BLens, XFL, SymLM, AsmDepictor, HexT5
- [data]`data/embedding/varclrCache`: Pre-computed VarCLR embeddings
- [data]`data/tokenizer/`: Tokenizers for both cross-binary and cross-project setting
- [code]`code/evaluation/`: Source code to evaluate BLens and other methods based on logs
- [code]`code/configs/`: Configuration files for BLens hyperparameters
- [code]`code/`: Source code to train new BLens models and evaluate them
- [document]`INSTALL.md`: Instructions to install the environment
- [document]`requirements.txt`: Packages required to install the environment
- [document]`README.md`: Instructions to use the artifact
- [document]`appendices/`: Details about the implementation and the strict experimental setting.

**Citation**:
```bibtex
@inproceedings{blens2025,
  title     = {{BLens}: Contrastive Captioning of Binary Functions using Ensemble Embedding},
  author    = {Benoit, Tristan and Wang, Yunru and Dannehl, Moritz and Kinder, Johannes},
  booktitle = {34th USENIX Security Symposium (USENIX Security 2025)},
  year      = {2025},
  publisher = {USENIX Association}
}
```

For the artifact citation, please refer to the [Zenodo record](https://doi.org/10.5281/zenodo.14713022).

\*: Equal contributions

## Data Directory

Pre-computed embeddings, datasets, tokenizers and logs are contained in the `data` folder from the corresponding [Zenodo record](https://doi.org/10.5281/zenodo.14713022).
Ensure you download and extract this folder before running the code.

## System Requirements

- **Storage**: Minimum 20 GB of available space. Each BLens experiment produces up to 50 GB of model weights.
- **Recommended Hardware**: 30 GB of memory and a GPU with 80 GB of VRAM.

## Installation Instructions

Please follow the detailed installation instructions provided in [INSTALL.md](INSTALL.md).


## Usage

### Reproduce the results of BLens and Other Methods


The original outputs are provided as logs, and the experimental results can be reproduced using the `evaluator.py` script.

```bash
workon blens
cd evaluation
python3 evaluator_all.py -data-dir=<PATH-TO-DATA-FOLDER> 
```
The results will be printed to stdout, with each table explicitly labeled to indicate its corresponding table in the paper.

### Training New Models

To initiate training for new models, use the `RunExp.py` script. Specify the experiment setting (cross-binary or cross-project). You also specify a sub-dataset obtained from the pre-processing of SymLM and AsmDepictor.

```bash
workon blens
CUDA_VISIBLE_DEVICES=0 python3 RunExp.py -data-dir=<PATH-TO-DATA-FOLDER> -d=test --cross-binary --symlm-subdataset -pretrain
```

This creates an experiment folder with the name `test`, inside `PATH-TO-DATA-FOLDER/xp/`.
Once initiated, the decoder fine-tuning can take place:

```bash
CUDA_VISIBLE_DEVICES=0 python3 RunExp.py --data-dir=<PATH-TO-DATA-FOLDER> -d=test --cross-binary --symlm-subdataset -train
```
Both the pre-training and fine-tuning phases last for 200 epochs. Every 10 epochs during fine-tuning, confidence thresholds are evaluated to find the optimal threshold according to the F1 score on the validation set. The model is then saved along with its threshold. Note that in the ablation study, each phase has 80 epochs, and we obtain a confidence threshold as well as a model every four epochs.

To infer function names on the test set with the best model on the validation set:

```bash
CUDA_VISIBLE_DEVICES=0 python3 RunExp.py -data-dir=<PATH-TO-DATA-FOLDER> -d=test --cross-binary --symlm-subdataset -inferBest
```

To execute the entire process in one run:

```bash
CUDA_VISIBLE_DEVICES=0 python3 RunExp.py -data-dir=<PATH-TO-DATA-FOLDER> -d=test --cross-binary --symlm-subdataset -pretrain -train -inferBest
```

The script provides a few more options:

- **Hyperparameters configuration**: Specify with `-config`a configuation filename (e.g., `ablation-simple.json`) inside the `configs` folder to set hyperparameters such as the the usage of COMBO pre-training and the LORD decoder, the number of epochs and more.
- **Resume training**: Use `-loadEpoch` to continue training from a specific epoch.
- **Inference without a threshold**: Add `-inferT0` for inferring function names without applying a threshold and `-reinferAllNoThreshold` for doing that after training a model with a threshold.
- **Inference with COMBO native decoder**: Add `-inferCOMBO` to infer names using the COMBO native decoder.

---

## Evaluating New Models

To evaluate new models, use the `new_experiments_evaluator.py` script.

```bash
workon blens
cd evaluation
python3 new_experiments_evaluator.py -data-dir=<PATH-TO-DATA-FOLDER> -d=test --cross-binary --symlm-subdataset
```

The script has two options related to the ablation study:

- **Evaluate models without a threshold**: Add `--evaluateNoThreshold` to evaluate a model without the threshold.
- **Evaluate COMBO native decoder models**: Add `--evaluateComboDecoder` to evaluate a model using the COMBO native decoder.

--- 

## License 

Copyright (C) 2024-2025 by Tristan Benoit, Yunru Wang, Moritz Dannehl, and Johannes Kinder.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

A copy of the GNU General Public License is included in the file LICENSE along with this program. 
