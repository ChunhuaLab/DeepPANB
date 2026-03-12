import torch
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import pandas as pd
import pyarrow.parquet as pq
from torch_geometric.data import Batch
from utils import extract_features, build_protein_graph
from concurrent.futures import as_completed
import time
from filelock import FileLock
import multiprocessing


def rebuild_meta(cache_dir):
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    # Collect all protein pt files (excluding meta file)
    pt_files = [f.stem for f in cache_path.glob("*.pt") if f.name != "dataset_meta.pt"]
    pt_files_sorted = sorted(pt_files)

    # Generate metadata
    meta = {
        'num_samples': len(pt_files_sorted),
        'sample_names': pt_files_sorted
    }

    # Save metadata
    torch.save(meta, cache_path / "dataset_meta.pt")
    print(f"Metadata rebuilt successfully with {len(pt_files_sorted)} samples")
    return meta


def load_dataset(txt_path: str, pdb_folder: str, device: torch.device,
                 use_cache: bool = True, max_workers: int = 8) -> list:
    processed_cache_dir = Path(os.environ.get("CACHE_DIR", "./processed_cache")) / Path(txt_path).stem
    status_file = processed_cache_dir / "processing_status.csv"
    meta_file = processed_cache_dir / "dataset_meta.pt"

    # ========== Cache integrity check ==========
    def check_cache_integrity():
        """Check the integrity of the cache directory"""
        if not meta_file.exists():
            return False, "Metadata file does not exist"

        try:
            meta = torch.load(meta_file, map_location='cpu')
        except Exception as e:
            return False, f"Metadata file corrupted: {str(e)}"

        # Check that every sample in metadata has a corresponding pt file
        for name in meta['sample_names']:
            pt_file = processed_cache_dir / f"{name}.pt"
            if not pt_file.exists():
                return False, f"Missing {name}.pt file"

        return True, "Cache is complete"

    # ========== Load or create cache ==========
    processed_cache_dir.mkdir(parents=True, exist_ok=True)

    # If cache is enabled and complete, load directly
    if use_cache:
        cache_ok, reason = check_cache_integrity()
        if cache_ok:
            print(f"Cache loaded: {processed_cache_dir}")
            return load_processed_dataset(processed_cache_dir)
        else:
            print(f"Cache incomplete: {reason}, starting processing...")
    else:
        print("Cache disabled, processing all samples...")

    # ========== Initialize processing status ==========
    if status_file.exists():
        status_df = pd.read_csv(status_file)
    else:
        status_df = pd.DataFrame(columns=["name", "status", "timestamp", "error"])

    # Use a container to wrap status_df so it can be modified in nested functions
    status_df_container = [status_df]

    # ========== Data preparation ==========
    proteins_df = parse_and_cache_fasta(txt_path, Path(os.environ.get("CACHE_DIR", "./processed_cache")) / "fasta_cache")
    pdb_dir = Path(pdb_folder)
    existing_files = {f.stem: f for f in pdb_dir.glob("*.pdb")}
    valid_names = [name for name in proteins_df.index if name in existing_files]

    # ========== Generate todo list ==========
    def get_todo_list():
        """Dynamically generate the list of samples to process"""
        nonlocal status_df_container
        status_df = status_df_container[0]  # Get current status from container

        todo = []
        for name in valid_names:
            pdb_path = existing_files[name]
            pdb_mtime = pdb_path.stat().st_mtime

            # Check if a successfully processed cache file already exists
            cache_file = processed_cache_dir / f"{name}.pt"
            if cache_file.exists():
                # If cache file is newer, skip processing
                cache_mtime = cache_file.stat().st_mtime
                if cache_mtime > pdb_mtime:
                    # Update status to cached (if not already recorded)
                    if name not in status_df['name'].values:
                        new_status = pd.DataFrame([[name, "cached", time.time(), ""]],
                                                  columns=["name", "status", "timestamp", "error"])
                        status_df = pd.concat([status_df, new_status], ignore_index=True)
                        # Update value in container
                        status_df_container[0] = status_df
                    continue

            # Three cases requiring processing:
            # 1. Never processed
            # 2. Previously failed
            # 3. PDB file has been updated
            status = status_df[status_df["name"] == name]
            if status.empty or \
                    (status["status"].iloc[0] == "failed") or \
                    (status["timestamp"].iloc[0] < pdb_mtime):
                todo.append(name)
        return todo

    # ========== Process all proteins in one pass ==========
    todo_list = get_todo_list()
    if todo_list:
        print(f"\nStarting processing | Pending samples: {len(todo_list)}")

        # Parallel processing logic
        with ProcessPoolExecutor(
                max_workers=max_workers,
                mp_context=multiprocessing.get_context('spawn')
        ) as executor:
            args_list = [
                (name, str(existing_files[name]), proteins_df.at[name, 'labels'])
                for name in todo_list
            ]
            future_to_name = {executor.submit(_process_single_wrapper, args): args[0] for args in args_list}

            # Progress bar and result collection
            progress = tqdm(as_completed(future_to_name), total=len(todo_list), desc="Processing")
            success_count = 0
            failed_count = 0

            for future in progress:
                name = future_to_name[future]
                try:
                    result = future.result()
                    if result is not None:
                        # Atomically save result
                        with FileLock(str(processed_cache_dir / f"{name}.lock")):
                            torch.save(result, processed_cache_dir / f"{name}.pt")
                            # Update status
                            new_status = pd.DataFrame([[name, "success", time.time(), ""]],
                                                      columns=["name", "status", "timestamp", "error"])
                            status_df = pd.concat([status_df, new_status], ignore_index=True)
                            status_df.to_csv(status_file, index=False)
                            success_count += 1
                except Exception as e:
                    error_msg = f"{type(e).__name__}: {str(e)}"
                    new_status = pd.DataFrame([[name, "failed", time.time(), error_msg]],
                                              columns=["name", "status", "timestamp", "error"])
                    status_df = pd.concat([status_df, new_status], ignore_index=True)
                    status_df.to_csv(status_file, index=False)
                    failed_count += 1

                # Update progress bar status
                progress.set_postfix_str(f"Success: {success_count}, Failed: {failed_count}", refresh=False)

        # Update metadata after processing
        rebuild_meta(processed_cache_dir)
        print(f"Processing complete | Success: {success_count}, Failed: {failed_count}")

    # ========== Final validation ==========
    # Rebuild metadata to ensure completeness
    meta = rebuild_meta(processed_cache_dir)

    # Check for missing samples
    missing = [name for name in valid_names if name not in meta['sample_names']]
    if missing:
        print(f"\n{len(missing)} samples still failed to process:")
        print("\n".join(missing[:5]) + ("..." if len(missing) > 5 else ""))
        raise RuntimeError("Some samples could not be processed, please check the logs")
    else:
        print(f"All samples processed | Total samples: {len(meta['sample_names'])}")

    return load_processed_dataset(processed_cache_dir)


def _process_single_wrapper(args):
    try:
        result = process_single_protein(*args)
        # Add memory cleanup
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        return result
    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        print(f"Processing failed: {args[0]} | Error: {error_msg}")
        return None


def process_single_protein(name: str, pdb_path: str, labels: str):
    device = torch.device('cpu')
    x, y, pos = extract_features(pdb_path, labels, device)

    # Add cutoff parameter (consistent with the cutoff value in the model)
    data = build_protein_graph(pdb_path, x, y, pos, device, cutoff=8.0)  # Assuming model cutoff is 8A
    return data


def parse_and_cache_fasta(txt_path, cache_dir):
    # Create cache directory
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    # Generate unique cache filename
    original_filename = Path(txt_path).stem  # Get filename without extension
    cache_path = os.path.join(cache_dir, f"{original_filename}.parquet")
    # Return cached file directly if it exists
    if os.path.exists(cache_path):
        return pd.read_parquet(cache_path)
    """Parse FASTA and cache as Parquet"""
    entries = []
    current_entry = {}

    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith('>'):
                if current_entry:
                    # ====== Length validation ======
                    seq_len = len(current_entry['sequence'])
                    label_len = len(current_entry['labels'])
                    if seq_len != label_len:
                        print(f"Skipping entry: {current_entry['name']} | Sequence length: {seq_len} | Label length: {label_len}")
                    else:
                        entries.append(current_entry)
                current_entry = {'name': line[1:].split()[0], 'sequence': '', 'labels': ''}
            else:
                if set(line) <= {'0', '1'}:
                    current_entry['labels'] += line
                else:
                    current_entry['sequence'] += line

        # Process the last entry
        if current_entry:
            if len(current_entry['sequence']) != len(current_entry['labels']):
                print(f"Warning: entry {current_entry['name']} has mismatched sequence and label lengths, skipped")
            else:
                entries.append(current_entry)

    # Convert to DataFrame and save
    # Optimize index and data types
    df = pd.DataFrame(entries).set_index('name', verify_integrity=True, drop=False)
    df['sequence'] = df['sequence'].astype('string[pyarrow]')
    df['labels'] = df['labels'].astype('string[pyarrow]')

    # Efficient storage
    df.to_parquet(cache_path, engine='pyarrow', compression='zstd')
    return df


def save_processed_dataset(dataset: list, cache_dir: Path):
    """Generate metadata based on dataset contents"""
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Save all samples
    sample_names = []
    for data in dataset:
        file_path = cache_dir / f"{data.name}.pt"
        torch.save(data, file_path)
        sample_names.append(data.name)

    # Generate precise metadata
    meta = {
        'num_samples': len(dataset),
        'sample_names': sample_names
    }

    # Atomic save
    temp_path = cache_dir / "dataset_meta.tmp"
    final_path = cache_dir / "dataset_meta.pt"
    torch.save(meta, temp_path)
    os.replace(temp_path, final_path)


def load_processed_dataset(cache_dir: Path) -> list:
    """Load the processed dataset"""
    try:
        meta = torch.load(cache_dir / "dataset_meta.pt", map_location='cpu')
        return [torch.load(cache_dir / f"{name}.pt") for name in meta['sample_names']]
    except FileNotFoundError:
        return []


def transfer_to_device(batch):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return Batch.from_data_list(batch).to(device)
