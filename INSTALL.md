# Installation Instructions

This document provides a step-by-step guide to set up the necessary environment and dependencies for BLens.

If you encounter any issues during the installation, we advice you to consult each tool's troubleshooting documentation.
If additional problems arise, the `requirements.txt` file may offer guidance, and feel free to reach out for further assistance.

## Prerequisites


Before proceeding, ensure you have `pip` installed. For further instructions on installing `pip`, please refer to the [official documentation](https://pip.pypa.io/en/stable/installation/).


If you have not installed `virtualenvwrapper` and `enchant`, you can do so with the following command:


```bash
sudo apt install virtualenv enchant
pip3 install virtualenvwrapper
```


Add the following lines to your shell startup file (e.g., `.bashrc`, `.profile`) to configure where virtual environments and development project directories should reside, and to source the `virtualenvwrapper` script:


```bash

# Virtualenvwrapper settings
export WORKON_HOME=$HOME/.virtualenvs
export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3
source ~/.local/bin/virtualenvwrapper.sh
```


After updating your shell startup file, reload it by running:


```bash
source ~/.bashrc
```


or the appropriate command for your shell.


## Create and Activate BLens Virtual Environment


```bash
mkvirtualenv blens
workon blens
pip install -r requirements.txt
python3 -m nltk.downloader words
python3 -m nltk.downloader stopwords

```


## Install VarCLR for Evaluation


### Install Rust (Linux / macOS)


If Rust is not already installed on your machine, use the following command to install it:


```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```


Follow the on-screen instructions to complete the installation and configure the cargo environment.


```bash
. "$HOME/.cargo/env"
```


### Clone and Install VarCLR


```bash

workon blens
git clone https://github.com/squaresLab/VarCLR.git
cd VarCLR

```


Edit `setup.py` so that the `transformers` package has no version requirement. Then install VarCLR with:


```bash
pip install -e .
```

