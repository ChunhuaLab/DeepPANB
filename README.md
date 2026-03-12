# DeepPANB: Integrating Protein Language Model with PaiNN Equivariant Graph Neural Networks for Prediction of Protein-Nucleic Acid Binding Sites

We develop a model called DeepPANB, which is an effective predictor based on the PaiNN E(3)-equivariant graph neural network integrated with the Ankh protein language model for protein-nucleic acid binding site prediction.

Authors: Jilong Zhang, ZhiXiang Wu, Jingjie Su, Xinyu Zhang, Yue Li, Zihan Li, Chunhua Li.

---

## Table of Contents

- [Requirements](#requirements)
- [Project Structure](#project-structure)
- [Quick Start: Predict on New Proteins](#quick-start-predict-on-new-proteins)
- [Full Pipeline](#full-pipeline)
  - [Step 1: Extract Ankh Features](#step-1-extract-ankh-features)
  - [Step 2: Extract IUPred Features](#step-2-extract-iupred-features)
  - [Step 3: Generate DSSP Files](#step-3-generate-dssp-files)
  - [Step 4: Train](#step-4-train)
- [Feature Dimensions](#feature-dimensions)
- [Model Architecture](#model-architecture)
- [Datasets](#datasets)
- [Citation](#citation)

---

## Requirements

```bash
pip install torch torch-geometric torch-scatter
pip install mdtraj biopython transformers
pip install pandas pyarrow tqdm filelock psutil scikit-learn
```

> Tested with Python 3.9+, PyTorch 2.x, CUDA 11.8+.

You also need:
- [**DSSP**](https://swift.cmbi.umcn.nl/gv/dssp/) installed and available on `PATH`
- [**IUPred2A**](https://iupred2a.elte.hu/) downloaded locally
- [**Ankh Large**](https://huggingface.co/ElnaggarLab/ankh-large) model downloaded locally

---

## Project Structure

```
DeepPANB/
├── README.md
├── final_model.pth              # Pretrained model weights (DNA-binding)
├── codes/
│   ├── predict.py               # Inference on new proteins  <- start here
│   ├── main.py                  # Training + cross-validation
│   ├── model.py                 # PaiNN model definition
│   ├── data_utils.py            # Dataset loading and caching
│   ├── utils.py                 # Feature extraction + graph building
│   ├── get_ankh_features.py     # Ankh Large embedding extraction
│   ├── get_dssp.py              # DSSP parsing utilities
│   └── get_iupred.py            # IUPred2A ANCHOR score extraction
└── example/
    ├── example_protein.pdb
    ├── ankh_features/
    │   └── example_protein.npy  # shape: [L, 1536]
    └── iupred_features/
        └── example_protein.npy  # shape: [L, 1]
```

---

## Quick Start: Predict on New Proteins

### 1. Prepare features

Pre-compute Ankh and IUPred features for your PDB files (see [Full Pipeline](#full-pipeline)).

To quickly test with the provided example:
```bash
python codes/predict.py \
    --pdb_dir    ./example \
    --ankh_dir   ./example/ankh_features \
    --iupred_dir ./example/iupred_features \
    --model_path ./final_model.pth \
    --output     ./predictions.csv
```

For your own proteins, organize files as:
```
my_proteins/
    proteinA.pdb
    proteinB.pdb

ankh_features/
    proteinA.npy      # shape: [L, 1536]
    proteinB.npy

iupred_features/
    my_proteins/      # subdirectory named after your PDB folder
        proteinA.npy  # shape: [L, 1]
        proteinB.npy
```

### 2. Run prediction

```bash
python codes/predict.py \
    --pdb_dir    ./my_proteins \
    --ankh_dir   ./ankh_features \
    --iupred_dir ./iupred_features \
    --model_path ./final_model.pth \
    --output     ./predictions.csv \
    --threshold  0.5
```

| Argument | Description | Default |
|---|---|---|
| `--pdb_dir` | Folder with `.pdb` files | required |
| `--ankh_dir` | Folder with Ankh `.npy` features | required |
| `--iupred_dir` | Folder with IUPred `.npy` features | required |
| `--model_path` | Path to `.pth` checkpoint | `./final_model.pth` |
| `--output` | Output CSV path | `./predictions.csv` |
| `--threshold` | Probability cutoff | `0.5` |

### 3. Output format

```
protein,residue_index,probability,predicted_label
proteinA,1,0.0312,0
proteinA,2,0.8741,1
...
```

---

## Full Pipeline

### Step 1: Extract Ankh Features

Download [Ankh Large](https://huggingface.co/ElnaggarLab/ankh-large) locally, then:

```bash
python codes/get_ankh_features.py
```

Edit `main()` in `get_ankh_features.py` to set:
- `ANKH_MODEL_PATH` — local path to Ankh Large (or set env var `ANKH_MODEL_PATH`)
- `PDB_FOLDERS` — list of PDB directories
- `OUTPUT_DIR` — output directory for `.npy` files

Output: one `.npy` per protein, shape `[L, 1536]`.

### Step 2: Extract IUPred Features

Download [IUPred2A](https://iupred2a.elte.hu/) and set `IUPRED_ROOT` in `get_iupred.py`, then:

```bash
python codes/get_iupred.py
```

Output: one `.npy` per protein, shape `[L, 1]` (ANCHOR score).

### Step 3: Generate DSSP Files

DSSP files are generated automatically during feature extraction. Make sure `dssp` is on your `PATH`:

```bash
dssp --version
```

### Step 4: Train

Set data paths via environment variables or edit `main.py` directly:

```bash
export TRAIN_TXT=./datasets/DNA-573_Train.fa
export TRAIN_PDB=./data/DNA_573_train
export TEST129_TXT=./datasets/DNA-129_Test.fa
export TEST129_PDB=./data/DNA_129_test
python codes/main.py
```

The `.txt` files use a FASTA-like format with binary binding labels:
```
>proteinID
MKTAYIAKQRQISFVKSHFSRQ...
00010011000110001000...
```

Training runs 5-fold cross-validation, then trains a final model on all data and evaluates on both test sets. Results are saved to `cv_results/`.

---

## Feature Dimensions

| Feature | Dim | Source |
|---|---|---|
| Amino acid one-hot | 20 | PDB sequence |
| Physicochemical (hydrophobicity, pI, volume, etc.) | 8 | Lookup tables |
| Nucleotide pairwise propensity | 3 | Derived from large-scale datasets |
| DSSP (sin/cos angles, ASA, secondary structure) | 13 | DSSP |
| SASA + charge | 2 | MDTraj / lookup |
| Ankh Large embedding | 1536 | Ankh Large PLM |
| IUPred ANCHOR score | 1 | IUPred2A |
| **Total** | **1583** | |

---

## Model Architecture

PaiNN E(3)-equivariant graph neural network over a protein residue graph:
- Nodes: C-alpha atoms with 1583-dim per-residue features
- Edges: all C-alpha pairs within 8 Angstrom cutoff; edge feature = log(seq_dist) / spatial_dist
- Message passing: coupled scalar + vector channels with radial basis functions and cosine cutoff
- Update: equivariant update with U/V vector transforms
- Output: per-residue sigmoid binding probability


## Help

For any questions, please contact us by chunhuali@bjut.edu.cn.
