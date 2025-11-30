
# Installation Guide for BLens pretrained models

This guide explains how to install BLens pretrained models on top of DEXTER, PalmTree, and CLAP.

You need sudo privileges, internet access, and approximately 10–15 GB free disk space.

## 0. Install Conda

If you don’t have Conda, follow the official instructions:

https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html

Alternatively, install Miniconda:

https://docs.conda.io/en/latest/miniconda.html

## 1. Install System Packages

Install dependencies:

```shell
sudo apt install zip tar g++ libpq-dev python3-dev graphviz libgraphviz-dev pkg-config openjdk-17-jdk  libffi7 libffi-dev git
```

## 2. Install radare2

```shell
wget https://github.com/radareorg/radare2/archive/refs/tags/5.5.4.zip -O radare2-5.5.4.zip
unzip radare2-5.5.4.zip
cd radare2-5.5.4
sudo bash sys/install.sh
```

## 3. Install PostgreSQL

```shell
sudo apt install postgresql
sudo locale-gen en_US en_US.UTF-8
sudo update-locale
```
Start/stop the PostgreSQL service:
```shell
sudo service postgresql start
sudo service postgresql stop
```

## 4. Install XFL

```shell
git clone https://github.com/lmu-plai/xfl.git
cd xfl/xfl-r
```

Configure the database (create DB, user, and schema):

```shell
sudo -u postgres psql
CREATE DATABASE xfl;
CREATE USER desyl;
ALTER USER desyl with password '123';
ALTER DATABASE xfl OWNER TO desyl;
GRANT ALL PRIVILEGES on DATABASE xfl TO desyl;
ALTER ROLE desyl WITH CREATEDB;
exit
sudo service postgresql restart
sudo -u postgres psql -d xfl -a -f XFL_DB.sql
```

Create and activate the XFL Conda environment (approximately 15 minutes):

```shell
conda env create -f XFL.yml
conda activate XFL
python -c "import nltk; nltk.download('words'); nltk.download('stopwords')"
```

## 5. Install Ghidra

```shell
wget https://github.com/NationalSecurityAgency/ghidra/releases/download/Ghidra_10.4_build/ghidra_10.4_PUBLIC_20230928.zip
sudo unzip ghidra_10.4_PUBLIC_20230928.zip -d /opt/
sudo mv /opt/ghidra_10.4_PUBLIC /opt/ghidra
rm ghidra_10.4_PUBLIC_20230928.zip
```

If you installed Ghidra somewhere other than /opt/, update the path in xfl-r/XFL/ghidra.py:

```
self.ghidraSupportDir = "/opt/ghidra/support/" # Set Ghidra support directory
```

## 6. Install BLens

```shell
git clone https://github.com/lmu-plai/blens.git
cd blens
```
Create and activate the BLens Conda environment (approximately 10 minutes):

```shell
conda env create -f BLens.yml
conda activate BLens
python -c "import nltk; nltk.download('words'); nltk.download('stopwords')"
```

## 7. Download Weights and Tables

Download and extract pretrained_data.tar.gz from the [Zenodo record](https://doi.org/10.5281/zenodo.14713022).

This will create a special BLens data folder (separate from the repository).

Restore the database with known library prototypes (required for DEXTER):

```shell
export PGPASSWORD='123' # Use the password you set earlier
dropdb -h localhost -U desyl -p 5432 xfl || true
pg_restore --create -h localhost -U desyl -d postgres -p 5432 data/Tables/xfl_blens.pgsql -v

```

Move the contents of data/res into the XFL resources directory:

```shell
# Adjust paths as needed
cp -r data/res/* /path/to/XFL/xfl-r/res/
```

PalmTree weights:
- Download from https://github.com/palmtreemodel/PalmTree/tree/master
- Place pretrained_palmtree and vocab files under data/palmtree/

## 8. Inference

Edit blens/Infer.sh and update the following variables to match your environment:

- TARGET: Path to a binary or a folder containing binaries to analyze

- XFL: Path to the XFL/xfl-r folder

- IDAB: Path to the IDA Pro 64-bit binary (idat64)

- BLENS: Path to the BLens folder

- BLENS_DATA: Path to the BLens data folder

Run the inference:
```shell
bash Infer.sh
```

After DEXTER, CLAP, PalmTree and BLens stages, the script outputs one line per function:
* binary_path virtual_address original_function_name -> BLens_predicted_name
