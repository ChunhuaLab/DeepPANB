import subprocess
import numpy as np
from pathlib import Path
import os
from tqdm import tqdm
import tempfile
from Bio.PDB import PDBParser
from utils import three_to_one
from concurrent.futures import ProcessPoolExecutor, as_completed


def extract_pdb_sequence(pdb_path: str) -> str:
    """Extract protein sequence from PDB file (enhanced error handling)"""
    try:
        parser = PDBParser(QUIET=True)  # Suppress warning output
        structure = parser.get_structure("protein", pdb_path)
        seq = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.id[0] != " ":
                        continue
                    resname = residue.resname
                    seq.append(three_to_one(resname))
        return "".join(seq)
    except Exception as e:
        raise ValueError(f"Cannot parse PDB file {pdb_path}: {str(e)}")


def process_single_pdb(pdb_path: str, output_dir: Path):
    """Process a single PDB file (optimized temp file handling)"""
    fasta_path = None
    try:
        pdb_name = Path(pdb_path).stem
        seq = extract_pdb_sequence(pdb_path)

        # Create temp file
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            fasta_path = f.name
            f.write(f">{pdb_name}\n{seq}\n")

        # Run IUPred2a (with timeout)
        def run_iupred_anchor(timeout=30):
            script_path = shlex.quote(str(IUPRED_ROOT / 'iupred2a.py'))
            safe_fasta = shlex.quote(fasta_path)

            # Run once using long mode + ANCHOR parameter
            cmd = (
                f"python3 {script_path} "
                f"-a {safe_fasta} long"  # Use long mode to get ANCHOR results simultaneously
            )

            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(IUPRED_ROOT)
            )
            return result

        anchor_result = run_iupred_anchor()

        # Parse results (enhanced fault tolerance)
        def parse_anchor_output(output):
            scores = []
            for line in output.splitlines():
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.split()
                if len(parts) >= 4:  # ANCHOR score is in the 4th column
                    try:
                        scores.append(float(parts[3]))  # Take only ANCHOR score
                    except ValueError:
                        continue
            return scores

        anchor_scores = parse_anchor_output(anchor_result.stdout)

        # Validate length
        seq_len = len(seq)
        if len(anchor_scores) != seq_len:
            raise ValueError(
                f"Prediction length mismatch: sequence length {seq_len}, ANCHOR {len(anchor_scores)}"
            )

        # Save features (ANCHOR only)
        features = np.array(anchor_scores).reshape(-1, 1)  # Changed from 3 columns to 1 column
        np.save(output_dir / f"{pdb_name}.npy", features)

    except subprocess.TimeoutExpired:
        print(f"Timeout, skipping: {pdb_path}")
    except Exception as e:
        print(f"Processing failed {pdb_path}: {str(e)}")
    finally:
        # Ensure temp file cleanup
        if fasta_path and os.path.exists(fasta_path):
            os.remove(fasta_path)
        # Clean up any leftover result files
        if fasta_path:
            result_glob = Path(fasta_path).parent / "*.result"
            for f in Path(fasta_path).parent.glob("*.result"):
                try:
                    os.remove(f)
                except:
                    pass


def batch_process_iupred(pdb_folder: str, output_root: str = "./iupred_features"):
    """Batch process an entire PDB folder"""
    pdb_dir = Path(pdb_folder)
    dataset_name = pdb_dir.name
    output_dir = Path(output_root) / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get list of files to process
    processed = {f.stem for f in output_dir.glob("*.npy")}
    todo = [f for f in pdb_dir.glob("*.pdb") if f.stem not in processed]

    print(f"Starting dataset processing: {dataset_name}")
    print(f"Total PDB files: {len(list(pdb_dir.glob('*.pdb')))}")
    print(f"Already processed: {len(processed)}, Pending: {len(todo)}")

    # Use multiprocessing for acceleration
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = []
        for pdb_file in todo:
            futures.append(
                executor.submit(
                    process_single_pdb,
                    str(pdb_file),
                    output_dir
                )
            )

        # Progress bar
        for future in tqdm(as_completed(futures), total=len(todo), desc="Generating IUPred features"):
            try:
                future.result()
            except Exception as e:
                print(f"Processing failed: {str(e)}")


if __name__ == "__main__":
    # Preprocess all datasets
    datasets = [
        "./data/DNA_573_train",
        "./data/DNA_129_test",
        "./data/DNA_Test_181"
    ]

    for dataset in datasets:
        batch_process_iupred(dataset)
