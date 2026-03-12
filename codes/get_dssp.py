import mdtraj as md
from Bio.PDB import PDBParser
import numpy as np
import torch
from torch_geometric.data import Data
import pandas as pd
import os
from pathlib import Path
import subprocess
from Bio import pairwise2

def process_dssp(dssp_file, chain_id):
    """Parse DSSP file and extract features (enhanced format compatibility)"""

    SS_TYPES = {'H': 0, 'B': 1, 'E': 2, 'G': 3, 'I': 4, 'T': 5, 'S': 6, ' ': 7}

    # Correct rASA_std definition (must match the previous declaration)
    rASA_std = [
        115, 225, 160, 150, 135, 180, 190, 75, 195, 175,
        170, 200, 185, 210, 145, 115, 140, 255, 230, 155,
        100, 100, 100, 100, 100, 100
    ]

    try:
        with open(dssp_file, 'r') as f:
            lines = [line.rstrip('\n') for line in f if line.strip()]
    except FileNotFoundError:
        raise ValueError(f"DSSP file does not exist: {dssp_file}")

    # More flexible start line detection
    header_patterns = [
        '  #  RESIDUE',
        ' # RESIDUE',
        '#  RESIDUE'
    ]

    # Find the first line containing "RESIDUE"
    start_idx = None
    for i, line in enumerate(lines):
        if any(pattern in line for pattern in header_patterns):
            start_idx = i + 1  # Data starts from the next line
            break

    if start_idx is None:
        # Output first 10 lines for debugging
        sample = '\n'.join(lines[:10])
        raise ValueError(
            f"Unexpected DSSP file format: {dssp_file}\n"
            f"Data start line not found, file header:\n{sample}"
        )

    # Skip blank lines when extracting features
    dssp_seq = []
    dssp_features = []
    for line in lines[start_idx:]:
        if len(line) < 115:
            continue

        chain = line[11]
        if chain != chain_id:
            continue

        aa = line[13].upper()  # Normalize to uppercase
        if aa == '!' or aa not in 'ACDEFGHIKLMNPQRSTVWY':
            continue  # Skip invalid residues

        # Normalized ASA calculation
        aa_idx = ord(aa) - ord('A')
        if aa_idx >= len(rASA_std):
            aa_idx = 0  # Use Alanine value if out of range
        max_asa = rASA_std[aa_idx]

        # Secondary structure
        ss = line[16]
        ss_onehot = [0] * 8
        ss_onehot[SS_TYPES.get(ss, 7)] = 1

        # Physicochemical features
        phi = float(line[103:109].strip() or 360)
        psi = float(line[109:115].strip() or 360)
        asa = float(line[34:38].strip() or 0)

        # Normalization
        norm_asa = min(asa / max_asa, 1.0)

        dssp_seq.append(aa)
        dssp_features.append([phi, psi, norm_asa] + ss_onehot)

    return ''.join(dssp_seq), np.array(dssp_features)


def align_dssp_features(pdb_seq, dssp_seq, dssp_features):
    """Align DSSP features with PDB sequence"""
    if pdb_seq == dssp_seq:
        return dssp_features

    # Global sequence alignment
    alignments = pairwise2.align.globalxx(pdb_seq, dssp_seq)
    aligned_pdb, aligned_dssp = alignments[0].seqA, alignments[0].seqB

    # Generate aligned features
    aligned_features = []
    dssp_ptr = 0
    for aa_pdb, aa_dssp in zip(aligned_pdb, aligned_dssp):
        if aa_pdb == '-':
            continue  # Skip gaps in PDB

        if aa_dssp == '-':
            # Pad missing features
            aligned_features.append([360, 360, 0] + [0] * 8)
        else:
            aligned_features.append(dssp_features[dssp_ptr])
            dssp_ptr += 1

    return np.array(aligned_features)
