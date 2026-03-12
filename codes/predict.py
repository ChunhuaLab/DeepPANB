"""
predict.py — Inference script for DNA-binding site prediction.

Usage:
    python predict.py --pdb_dir ./my_proteins \
                      --ankh_dir ./ankh_features \
                      --iupred_dir ./iupred_features \
                      --model_path ./final_model.pth \
                      --output ./predictions.csv \
                      --threshold 0.5
"""

import argparse
import os
import csv
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

from model import PAINN
from utils import extract_features, build_protein_graph


def load_model(model_path: str, device: torch.device) -> PAINN:
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    params = ckpt['params']
    model = PAINN(
        input_dim=1583,
        hidden_dim=params['hidden_dim'],
        num_layers=params['num_layers'],
        dropout_rate=params['dropout_rate']
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Model loaded from {model_path}")
    print(f"  hidden_dim={params['hidden_dim']}, num_layers={params['num_layers']}, "
          f"dropout_rate={params['dropout_rate']}")
    return model


def predict_single(pdb_path: str, model: PAINN, device: torch.device,
                   ankh_dir: str, iupred_dir: str, threshold: float):
    """
    Run prediction on a single PDB file.
    Returns (residue_probs, residue_labels, sequence_length).
    """
    pdb_name = Path(pdb_path).stem

    # Build a dummy label string (all zeros) — labels are not needed for inference
    import mdtraj as md
    traj = md.load(pdb_path)
    num_residues = traj.topology.n_residues
    dummy_labels = '0' * num_residues

    x, y, pos = extract_features(
        pdb_path, dummy_labels, torch.device('cpu'),
        ankh_features_dir=ankh_dir,
        iupred_features_dir=iupred_dir
    )

    data = build_protein_graph(pdb_path, x, y, pos, torch.device('cpu'), cutoff=8.0)
    data = data.to(device)

    with torch.no_grad():
        probs = model(data).squeeze().cpu().numpy()

    if probs.ndim == 0:
        probs = probs.reshape(1)

    labels = (probs > threshold).astype(int)
    return probs, labels, num_residues


def main():
    parser = argparse.ArgumentParser(
        description="Predict DNA-binding residues from PDB files."
    )
    parser.add_argument('--pdb_dir', required=True,
                        help='Directory containing input .pdb files')
    parser.add_argument('--ankh_dir', required=True,
                        help='Directory containing pre-computed Ankh features (.npy)')
    parser.add_argument('--iupred_dir', required=True,
                        help='Directory containing pre-computed IUPred features (.npy)')
    parser.add_argument('--model_path', default='./final_model.pth',
                        help='Path to trained model checkpoint (default: ./final_model.pth)')
    parser.add_argument('--output', default='./predictions.csv',
                        help='Output CSV file path (default: ./predictions.csv)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Probability threshold for positive prediction (default: 0.5)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = load_model(args.model_path, device)

    pdb_dir = Path(args.pdb_dir)
    pdb_files = sorted(pdb_dir.glob('*.pdb'))
    if not pdb_files:
        print(f"No .pdb files found in {args.pdb_dir}")
        return

    print(f"\nFound {len(pdb_files)} PDB files. Running predictions...\n")

    rows = []
    failed = []

    for pdb_file in pdb_files:
        try:
            probs, labels, n_res = predict_single(
                str(pdb_file), model, device,
                args.ankh_dir, args.iupred_dir, args.threshold
            )
            n_binding = int(labels.sum())
            print(f"  {pdb_file.name}: {n_res} residues, "
                  f"{n_binding} predicted binding sites "
                  f"({n_binding/n_res*100:.1f}%)")

            for i, (p, l) in enumerate(zip(probs.tolist(), labels.tolist())):
                rows.append({
                    'protein': pdb_file.stem,
                    'residue_index': i + 1,
                    'probability': round(p, 4),
                    'predicted_label': int(l)
                })
        except Exception as e:
            print(f"  FAILED {pdb_file.name}: {e}")
            failed.append(pdb_file.name)

    # Write CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['protein', 'residue_index',
                                               'probability', 'predicted_label'])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone. Results saved to {output_path}")
    print(f"  Processed: {len(pdb_files) - len(failed)}/{len(pdb_files)}")
    if failed:
        print(f"  Failed: {failed}")


if __name__ == '__main__':
    main()
