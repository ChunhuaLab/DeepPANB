import mdtraj as md
from Bio.PDB import PDBParser
import numpy as np
import torch
from torch_geometric.data import Data
import pandas as pd
import os
from pathlib import Path
from get_dssp import process_dssp, align_dssp_features
import subprocess
import re


def three_to_one(aa):
    """Convert three-letter to one-letter amino acid code (enhanced compatibility)"""
    conversion = {
        'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
        'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
        'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
        'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
        'ASX': 'B', 'GLX': 'Z', 'SEC': 'U', 'PYL': 'O', 'XLE': 'J'  # Handle special residues
    }
    return conversion.get(aa.upper(), 'X')  # Return X for non-standard residues


# ... (keep all defined constants unchanged) ...

# Hydrophobicity (Kyte-Doolittle scale)
HYDROPHOBICITY = {
    'ALA': 1.8, 'ARG': -4.5, 'ASN': -3.5, 'ASP': -3.5, 'CYS': 2.5,
    'GLN': -3.5, 'GLU': -3.5, 'GLY': -0.4, 'HIS': -3.2, 'ILE': 4.5,
    'LEU': 3.8, 'LYS': -3.9, 'MET': 1.9, 'PHE': 2.8, 'PRO': -1.6,
    'SER': -0.8, 'THR': -0.7, 'TRP': -0.9, 'TYR': -1.3, 'VAL': 4.2
}

# Hydrophilicity (Hopp-Woods scale)
HYDROPHILICITY = {
    'ALA': -0.5, 'ARG': 3.0, 'ASN': 0.2, 'ASP': 3.0, 'CYS': -1.0,
    'GLN': 0.2, 'GLU': 3.0, 'GLY': 0.0, 'HIS': -0.5, 'ILE': -1.8,
    'LEU': -1.8, 'LYS': 3.0, 'MET': -1.3, 'PHE': -2.5, 'PRO': 0.0,
    'SER': 0.3, 'THR': -0.4, 'TRP': -3.4, 'TYR': -2.3, 'VAL': -1.5
}

# Volume (Zimmerman volume scale, unit A^3)
RESIDUE_VOLUME = {
    'ALA': 91.5, 'ARG': 202.0, 'ASN': 135.2, 'ASP': 124.5, 'CYS': 118.0,
    'GLN': 161.1, 'GLU': 155.1, 'GLY': 66.4, 'HIS': 167.3, 'ILE': 169.0,
    'LEU': 168.6, 'LYS': 171.3, 'MET': 170.8, 'PHE': 203.4, 'PRO': 129.3,
    'SER': 99.1, 'THR': 122.1, 'TRP': 237.6, 'TYR': 203.6, 'VAL': 141.7
}

# Isoelectric point (pI values)
ISOELECTRIC_POINT = {
    'ALA': 6.00, 'ARG': 10.76, 'ASN': 5.41, 'ASP': 2.77, 'CYS': 5.07,
    'GLN': 5.65, 'GLU': 3.22, 'GLY': 5.97, 'HIS': 7.59, 'ILE': 6.02,
    'LEU': 5.98, 'LYS': 9.74, 'MET': 5.74, 'PHE': 5.48, 'PRO': 6.30,
    'SER': 5.68, 'THR': 5.60, 'TRP': 5.89, 'TYR': 5.66, 'VAL': 5.96
}

DNA_PREFERENCE = {
    'ALA': {'mean': 0.8625, 'max': 1.02, 'min': 0.7},
    'ARG': {'mean': 1.805, 'max': 1.97, 'min': 1.43},
    'ASN': {'mean': 1.0275, 'max': 1.31, 'min': 0.77},
    'ASP': {'mean': 0.495, 'max': 0.56, 'min': 0.46},
    'CYS': {'mean': 0.7875, 'max': 1.05, 'min': 0.48},
    'GLN': {'mean': 0.8025, 'max': 0.97, 'min': 0.7},
    'GLU': {'mean': 0.4, 'max': 0.45, 'min': 0.34},
    'GLY': {'mean': 1.285, 'max': 1.54, 'min': 1.04},
    'HIS': {'mean': 1.1925, 'max': 1.44, 'min': 0.93},
    'ILE': {'mean': 0.9625, 'max': 1.01, 'min': 0.93},
    'LEU': {'mean': 0.575, 'max': 0.66, 'min': 0.42},
    'LYS': {'mean': 1.3725, 'max': 1.43, 'min': 1.26},
    'MET': {'mean': 0.975, 'max': 1.1, 'min': 0.82},
    'PHE': {'mean': 0.915, 'max': 1.16, 'min': 0.67},
    'PRO': {'mean': 0.755, 'max': 0.83, 'min': 0.62},
    'SER': {'mean': 1.2275, 'max': 1.39, 'min': 1.02},
    'THR': {'mean': 1.2075, 'max': 1.39, 'min': 0.95},
    'TRP': {'mean': 1.25, 'max': 1.55, 'min': 0.98},
    'TYR': {'mean': 1.3425, 'max': 1.5, 'min': 1.19},
    'VAL': {'mean': 0.85, 'max': 1.03, 'min': 0.6}
}

# Atom counts
ATOM_COUNTS = {
    'ALA': 5, 'ARG': 11, 'ASN': 8, 'ASP': 8, 'CYS': 6,
    'GLN': 9, 'GLU': 9, 'GLY': 4, 'HIS': 10, 'ILE': 8,
    'LEU': 8, 'LYS': 9, 'MET': 8, 'PHE': 11, 'PRO': 7,
    'SER': 6, 'THR': 7, 'TRP': 14, 'TYR': 12, 'VAL': 7
}

# Structural preference (Chou-Fasman parameters)
STRUCTURE_PREFERENCE = {
    'ALA': (1.42, 0.83, 0.66), 'ARG': (0.98, 0.93, 1.08),
    'ASN': (0.67, 0.89, 1.34), 'ASP': (1.01, 0.54, 1.46),
    'CYS': (0.70, 1.19, 1.11), 'GLN': (1.11, 1.10, 0.98),
    'GLU': (1.51, 0.37, 1.56), 'GLY': (0.57, 0.75, 1.56),
    'HIS': (1.00, 0.87, 1.24), 'ILE': (1.08, 1.60, 0.47),
    'LEU': (1.21, 1.30, 0.59), 'LYS': (1.16, 0.74, 1.01),
    'MET': (1.45, 1.05, 0.60), 'PHE': (1.13, 1.38, 0.60),
    'PRO': (0.57, 0.55, 1.52), 'SER': (0.77, 0.75, 1.43),
    'THR': (0.83, 1.19, 1.03), 'TRP': (1.08, 1.37, 0.96),
    'TYR': (0.69, 1.47, 1.14), 'VAL': (1.06, 1.70, 0.50)
}


def extract_features(pdb_path, label_str, device, ankh_features_dir="./ankh_features",
                     iupred_features_dir="./iupred_features"):
    """
    Extract protein features

    Args:
        pdb_path: Path to PDB file
        label_str: Label string
        device: Device
        ankh_features_dir: Path to Ankh features directory
        iupred_features_dir: Path to IUPred features directory
    """
    traj = md.load(pdb_path)
    num_residues = traj.topology.n_residues  # Get actual residue count
    num_labels = len(label_str.strip())

    if num_residues != num_labels:
        raise ValueError(f"PDB residue count ({num_residues}) does not match label count ({num_labels}): {pdb_path}")

    parser = PDBParser()
    structure = parser.get_structure("protein", pdb_path)

    # ========== Load pre-generated IUPred features ==========
    pdb_name = Path(pdb_path).stem
    dataset_name = Path(pdb_path).parent.name  # e.g. DNA_573_train

    # Use parameterized IUPred feature path
    iupred_feat_path = Path(iupred_features_dir) / dataset_name / f"{pdb_name}.npy"

    if iupred_feat_path.exists():
        iupred_features = np.load(iupred_feat_path)
        if len(iupred_features) != num_residues:
            # Try to adjust length to match
            if len(iupred_features) > num_residues:
                iupred_features = iupred_features[:num_residues]
                print(f"Warning: IUPred features truncated {len(iupred_features)} -> {num_residues} for {pdb_name}")
            else:
                # Pad with zeros
                padding_size = num_residues - len(iupred_features)
                if len(iupred_features.shape) == 1:
                    iupred_features = np.concatenate([iupred_features, np.zeros(padding_size)])
                else:
                    padding = np.zeros((padding_size, iupred_features.shape[1]))
                    iupred_features = np.concatenate([iupred_features, padding], axis=0)
                print(f"Warning: IUPred features padded {len(iupred_features) - padding_size} -> {num_residues} for {pdb_name}")
    else:
        print(f"Warning: IUPred feature file not found: {iupred_feat_path}, using zero padding")
        iupred_features = np.zeros(num_residues)

    # Ensure IUPred features are 1D (based on original code, this appears to be a 1D feature)
    if len(iupred_features.shape) > 1:
        iupred_features = iupred_features.flatten()[:num_residues]

    iupred_features_norm = (iupred_features - 0.5) * 2

    # ========== DSSP processing ==========
    dssp_file = Path(pdb_path).with_suffix('.dssp')
    if not dssp_file.exists():
        try:
            result = subprocess.run(
                ['dssp', pdb_path, dssp_file],
                timeout=10,  # Set 10-second timeout
                stderr=subprocess.PIPE
            )
            if result.returncode != 0:
                raise RuntimeError(f"DSSP generation failed: {result.stderr.decode()}")
        except subprocess.TimeoutExpired:
            raise RuntimeError("DSSP generation timed out, file may be too large")
        except FileNotFoundError:
            print("Warning: DSSP program not found, using zero padding")
            # If DSSP program is missing, use zero padding
            aligned_dssp = np.zeros((num_residues, 13))
            dssp_combined = aligned_dssp
        except Exception as e:
            print(f"Warning: DSSP processing failed {e}, using zero padding")
            aligned_dssp = np.zeros((num_residues, 13))
            dssp_combined = aligned_dssp

    if dssp_file.exists():
        try:
            # Parse DSSP features
            chain_id = Path(pdb_path).stem.split('_')[-1][0]  # Assumes filename format XXXX_A.pdb
            dssp_seq, dssp_features = process_dssp(dssp_file, chain_id)

            # Align features
            pdb_seq = ''.join([three_to_one(r.name) for r in traj.topology.residues])
            aligned_dssp = align_dssp_features(pdb_seq, dssp_seq, dssp_features)

            # Convert angle features
            angles = aligned_dssp[:, 0:2]
            angles_rad = np.deg2rad(angles)
            dssp_sin = np.sin(angles_rad)
            dssp_cos = np.cos(angles_rad)

            # Combine all DSSP features
            dssp_combined = np.hstack([
                dssp_sin,
                dssp_cos,
                aligned_dssp[:, 2].reshape(-1, 1),  # ASA
                aligned_dssp[:, 3:]  # SS one-hot
            ])  # Total dimensions 2+2+1+8=13
        except Exception as e:
            print(f"Warning: DSSP feature processing failed {e}, using zero padding")
            dssp_combined = np.zeros((num_residues, 13))

    # ========== Load Ankh features ==========
    pdb_name = Path(pdb_path).stem

    # Use parameterized Ankh feature path, matching the output path of the feature extraction code
    ankh_feat_path = Path(ankh_features_dir) / f"{pdb_name}.npy"

    if ankh_feat_path.exists():
        ankh_features = np.load(ankh_feat_path)  # [L, 1536] - Ankh Large feature dimension
        if len(ankh_features) != num_residues:
            # Adjust if lengths do not match
            if len(ankh_features) > num_residues:
                # Truncate to matching length
                ankh_features = ankh_features[:num_residues]
                print(f"Warning: Ankh features truncated {len(ankh_features)} -> {num_residues} for {pdb_name}")
            else:
                # Pad with zeros if Ankh features are too short
                padding_size = num_residues - len(ankh_features)
                padding = np.zeros((padding_size, ankh_features.shape[1]))
                ankh_features = np.concatenate([ankh_features, padding], axis=0)
                print(f"Warning: Ankh features padded {len(ankh_features) - padding_size} -> {num_residues} for {pdb_name}")
    else:
        print(f"Warning: Ankh feature file not found: {ankh_feat_path}, using zero padding")
        # Use 1536-dim zero vector as default feature
        ankh_features = np.zeros((num_residues, 1536))

    # ========== Move SASA calculation here (outside the loop) ==========
    try:
        sasa_values = md.shrake_rupley(traj, mode='residue')[0]
        if np.std(sasa_values) < 1e-6:  # Avoid division by zero
            sasa_norm = sasa_values - np.mean(sasa_values)  # Compute for all residues at once
        else:
            sasa_norm = (sasa_values - np.mean(sasa_values)) / np.std(sasa_values)  # Compute for all residues at once
        # Clip features
        sasa_norm = np.clip(sasa_norm, -5.0, 5.0)
    except Exception as e:
        print(f"Warning: SASA calculation failed {e}, using zeros")
        sasa_norm = np.zeros(num_residues)  # Create zero array matching residue count

    # Extract residue-level features
    features = []
    for i, residue in enumerate(traj.topology.residues):
        # =================== DNA preference features ===================
        # Get DNA preference features for the current amino acid
        aa_type = residue.name.upper()
        pref = DNA_PREFERENCE.get(aa_type, {'mean': 0.0, 'max': 0.0, 'min': 0.0})

        # Compute normalized features (optional)
        # Assuming all values are roughly in the 0-2 range, normalize to 0-1
        norm_mean = pref['mean'] / 2.0
        norm_max = pref['max'] / 2.0
        norm_min = pref['min'] / 2.0

        # Raw values can also be used
        dna_pref_features = [norm_mean, norm_max, norm_min]

        # ----------------------------
        # Feature 1: Amino acid type (one-hot encoding)
        # ----------------------------
        aa_types = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY',
                    'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER',
                    'THR', 'TRP', 'TYR', 'VAL']
        aa_onehot = [1 if aa_type == t else 0 for t in aa_types]

        # ----------------------------
        # Feature 2: Secondary structure (DSSP)
        # ----------------------------
        dssp_feature = dssp_combined[i].tolist()

        # ----------------------------
        # Feature 3: Solvent accessible surface area (SASA) - use pre-computed value directly
        # ----------------------------
        sasa_val = sasa_norm[i] if i < len(sasa_norm) else 0.0  # Simple index, no recomputation

        # ----------------------------
        # Features 4-8: Physicochemical features
        # ----------------------------
        hydro = HYDROPHOBICITY.get(aa_type, 0.0)

        # Charge (at pH 7)
        charge = {
            'ARG': 1, 'LYS': 1, 'ASP': -1, 'GLU': -1, 'HIS': 0.1,
            'ALA': 0, 'ASN': 0, 'CYS': 0, 'GLN': 0, 'GLY': 0,
            'ILE': 0, 'LEU': 0, 'MET': 0, 'PHE': 0, 'PRO': 0,
            'SER': 0, 'THR': 0, 'TRP': 0, 'TYR': 0, 'VAL': 0
        }
        charge_val = charge.get(aa_type, 0.0)

        # Hydrophilicity
        hydrophilic = HYDROPHILICITY.get(aa_type, 0.0)

        # Volume (normalized to 0-1)
        volume = RESIDUE_VOLUME.get(aa_type, 100.0)
        max_volume = max(RESIDUE_VOLUME.values())
        norm_volume = volume / max_volume

        # Isoelectric point (normalized to 0-1)
        pI = ISOELECTRIC_POINT.get(aa_type, 5.0)
        norm_pI = (pI - 2.5) / (10.76 - 2.5)  # Based on actual range 2.77-10.76

        # Atom count (normalized)
        atom_count = ATOM_COUNTS.get(aa_type, 5)
        norm_atoms = atom_count / 14.0  # Max atom count 14 (TRP)

        # Structural preference
        alpha_pref, beta_pref, coil_pref = STRUCTURE_PREFERENCE.get(aa_type, (1.0, 1.0, 1.0))

        # Combine physicochemical features
        physchem_features = [
            hydro,
            hydrophilic,
            norm_volume,
            norm_pI,
            norm_atoms,
            alpha_pref,
            beta_pref,
            coil_pref
        ]

        # Combine features (using Ankh features instead of ESM features)
        residue_features = (
                aa_onehot +  # 20
                physchem_features +  # 8
                dna_pref_features +  # 3 DNA preference features
                dssp_feature +  # 13
                [sasa_val, charge_val] +  # 2
                ankh_features[i].tolist() +
                [iupred_features_norm[i]]  # 1D IUPred feature
        )

        features.append(residue_features)

    # Label processing
    labels = [int(c) for c in label_str.strip()]

    # Convert to PyTorch Tensor
    x = torch.tensor(features, dtype=torch.float, device=device)
    y = torch.tensor(labels, dtype=torch.float, device=device).unsqueeze(1)  # Shape becomes (num_nodes, 1)

    # Extract CA atom coordinates
    ca_indices = [atom.index for atom in traj.topology.atoms if atom.name == 'CA']
    ca_coords = traj.xyz[0, ca_indices]  # [N_res, 3]

    # Convert to PyTorch Tensor
    pos = torch.tensor(ca_coords, dtype=torch.float, device=device)

    return x, y, pos  # Return features, labels, and coordinates


def build_protein_graph(pdb_path, x, y, pos, device, cutoff=15.0):
    """Build protein graph"""
    traj = md.load(pdb_path)

    # More robust CA atom selection
    ca_indices = traj.topology.select("protein and name CA")
    if len(ca_indices) == 0:
        raise ValueError(f"No CA atoms found for protein {Path(pdb_path).stem}")

    # Use general distance calculation
    ca_pairs = np.array([[i, j] for i in ca_indices for j in ca_indices if i < j])

    # Compute distances between all CA atom pairs
    distances = md.compute_distances(traj, ca_pairs, periodic=False)[0] * 10.0  # nm to Angstrom

    # Generate edge index
    edge_index = []
    edge_attr = []  # Edge feature list

    for k, (i, j) in enumerate(ca_pairs):
        if distances[k] <= cutoff:
            # Convert to residue index
            residue_i = traj.topology.atom(i).residue
            residue_j = traj.topology.atom(j).residue

            # Compute sequence distance (log absolute difference)
            seq_diff = np.log(np.abs(residue_i.index - residue_j.index) + 1e-3)  # +1e-3 to avoid log(0)

            # Spatial distance (in Angstrom)
            spatial_dist = distances[k]

            # Compute ratio feature
            ratio_feature = seq_diff / (spatial_dist + 1e-3)  # +1e-3 to avoid division by zero

            # Convert to node index
            idx_i = np.where(ca_indices == i)[0][0]
            idx_j = np.where(ca_indices == j)[0][0]

            # Add bidirectional edges and features (each edge added twice)
            edge_index.append([idx_i, idx_j])
            edge_index.append([idx_j, idx_i])
            edge_attr.append([ratio_feature])
            edge_attr.append([ratio_feature])  # Bidirectional edges share the same feature

    return Data(
        x=x,
        edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float, device=device),  # Edge features
        y=y,
        pos=pos,
        name=Path(pdb_path).stem
    )


def batch_extract_features(data_dir, ankh_features_dir="./ankh_features", iupred_features_dir="./iupred_features",
                           device="cuda"):
    """
    Helper function for batch feature extraction

    Args:
        data_dir: Data directory path
        ankh_features_dir: Path to Ankh features directory
        iupred_features_dir: Path to IUPred features directory
        device: Device
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise ValueError(f"Data directory does not exist: {data_dir}")

    pdb_files = list(data_path.glob("*.pdb"))
    if not pdb_files:
        raise ValueError(f"No PDB files found in {data_dir}")

    print(f"Starting processing of {len(pdb_files)} PDB files...")

    graphs = []
    failed_files = []

    for i, pdb_file in enumerate(pdb_files, 1):
        try:
            print(f"Progress: {i}/{len(pdb_files)} - {pdb_file.name}")

            # A corresponding label file is needed; adjust according to actual situation
            # Assuming label file has the same name as the PDB file but a different extension
            label_file = pdb_file.with_suffix('.label')  # Or another format
            if not label_file.exists():
                print(f"Warning: Label file not found {label_file}")
                continue

            with open(label_file, 'r') as f:
                label_str = f.read().strip()

            # Extract features
            x, y, pos = extract_features(
                str(pdb_file),
                label_str,
                device,
                ankh_features_dir=ankh_features_dir,
                iupred_features_dir=iupred_features_dir
            )

            # Build graph
            graph = build_protein_graph(str(pdb_file), x, y, pos, device)
            graphs.append(graph)

        except Exception as e:
            print(f"Processing failed: {pdb_file.name} - {str(e)}")
            failed_files.append(pdb_file.name)

    print(f"Processing complete! Success: {len(graphs)}, Failed: {len(failed_files)}")
    if failed_files:
        print(f"Failed files: {failed_files}")

    return graphs
