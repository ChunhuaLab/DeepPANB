import torch
import numpy as np
from pathlib import Path
from Bio.PDB import PDBParser
from transformers import AutoTokenizer, AutoModel
import os
import warnings
warnings.filterwarnings('ignore')

class FixedAnkhExtractor:
    def __init__(self, model_path, device=None):
        """
        Fixed ANKH feature extractor

        Args:
            model_path: Path to ANKH model
            device: Compute device
        """
        self.model_path = Path(model_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.max_seq_len = 1500  # Reduced max length

        print(f"Initializing ANKH feature extractor")
        print(f"Model path: {self.model_path}")
        print(f"Device: {self.device}")

        self._load_model()

    def _load_model(self):
        """Safely load model"""
        try:
            # Check model paths
            tokenizer_path = self.model_path / "tokenizer"
            model_path = self.model_path / "model"

            if not tokenizer_path.exists():
                raise FileNotFoundError(f"Tokenizer path does not exist: {tokenizer_path}")
            if not model_path.exists():
                raise FileNotFoundError(f"Model path does not exist: {model_path}")

            # Load tokenizer
            print("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(tokenizer_path),
                local_files_only=True
            )
            print(f"Tokenizer loaded successfully")

            # Load model - use float32 to avoid precision issues
            print("Loading model...")
            self.model = AutoModel.from_pretrained(
                str(model_path),
                local_files_only=True,
                torch_dtype=torch.float32,  # Force float32
                trust_remote_code=True
            )

            self.model.eval()
            self.model = self.model.to(self.device)

            print(f"Model loaded successfully")
            print(f"Model type: {type(self.model)}")
            print(f"Model device: {next(self.model.parameters()).device}")
            print(f"Model dtype: {next(self.model.parameters()).dtype}")

        except Exception as e:
            print(f"Model loading failed: {e}")
            raise e

    def extract_sequence_from_pdb(self, pdb_path):
        """Extract sequence from PDB file"""
        try:
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure("protein", pdb_path)

            # Three-letter to one-letter conversion
            aa_dict = {
                'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
                'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
                'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
                'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
                'MSE': 'M', 'SEC': 'U', 'PYL': 'O', 'ASX': 'B', 'GLX': 'Z'
            }

            sequence = []
            residue_count = 0

            for model in structure:
                for chain in model:
                    for residue in chain:
                        # Only process standard amino acid residues
                        if residue.id[0] == ' ':  # Standard residue
                            resname = residue.resname.strip().upper()
                            aa = aa_dict.get(resname, 'X')
                            sequence.append(aa)
                            residue_count += 1

            final_sequence = ''.join(sequence)

            print(f"Sequence extracted: {Path(pdb_path).stem}")
            print(f"   Residue count: {residue_count}")
            print(f"   Sequence length: {len(final_sequence)}")
            print(f"   First 20: {final_sequence[:20]}")

            if len(final_sequence) == 0:
                raise ValueError("Extracted sequence is empty")

            return final_sequence

        except Exception as e:
            print(f"Sequence extraction failed {Path(pdb_path).stem}: {e}")
            raise e

    def safe_tokenize(self, sequence):
        """Safe tokenization"""
        try:
            # Try multiple formats
            formats = [
                sequence,  # Raw sequence
                ' '.join(sequence),  # Space-separated
                f"<s> {' '.join(sequence)} </s>",  # With special tokens
            ]

            for i, formatted_seq in enumerate(formats):
                try:
                    # Truncate long sequences
                    if len(formatted_seq.split()) > self.max_seq_len:
                        print(f"Sequence too long, truncating to {self.max_seq_len} tokens")
                        tokens = formatted_seq.split()[:self.max_seq_len]
                        formatted_seq = ' '.join(tokens)

                    inputs = self.tokenizer(
                        formatted_seq,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=self.max_seq_len + 10
                    )

                    print(f"Tokenization successful (format {i+1})")
                    print(f"   Input shape: {inputs['input_ids'].shape}")
                    return inputs

                except Exception as e:
                    print(f"Format {i+1} failed: {e}")
                    continue

            raise ValueError("All tokenization formats failed")

        except Exception as e:
            print(f"Tokenization completely failed: {e}")
            raise e

    def safe_inference(self, inputs, original_sequence):
        """Safe model inference"""
        try:
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                # Try multiple inference methods
                embeddings = None

                # Method 1: Direct model call
                try:
                    outputs = self.model(**inputs)
                    if hasattr(outputs, 'last_hidden_state'):
                        embeddings = outputs.last_hidden_state[0]
                        print("Using last_hidden_state")
                    elif hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                        embeddings = outputs.hidden_states[-1][0]
                        print("Using hidden_states[-1]")
                    elif isinstance(outputs, tuple) and len(outputs) > 0:
                        embeddings = outputs[0][0]
                        print("Using outputs[0]")
                    else:
                        raise ValueError("Cannot extract embeddings from output")

                except Exception as e1:
                    print(f"Method 1 failed: {e1}")

                    # Method 2: Try encoder
                    try:
                        if hasattr(self.model, 'encoder'):
                            encoder_outputs = self.model.encoder(**inputs)
                            if hasattr(encoder_outputs, 'last_hidden_state'):
                                embeddings = encoder_outputs.last_hidden_state[0]
                                print("Using encoder.last_hidden_state")
                            else:
                                embeddings = encoder_outputs[0][0]
                                print("Using encoder outputs[0]")
                        else:
                            raise ValueError("Model has no encoder attribute")
                    except Exception as e2:
                        print(f"Method 2 failed: {e2}")
                        raise ValueError("All inference methods failed")

                # Check output validity
                if embeddings is None:
                    raise ValueError("embeddings is None")

                print(f"Raw output check:")
                print(f"   Shape: {embeddings.shape}")
                print(f"   dtype: {embeddings.dtype}")
                print(f"   Device: {embeddings.device}")
                print(f"   NaN count: {torch.isnan(embeddings).sum().item()}")
                print(f"   Inf count: {torch.isinf(embeddings).sum().item()}")

                # Handle NaN and Inf
                if torch.isnan(embeddings).any() or torch.isinf(embeddings).any():
                    print("NaN/Inf values found, cleaning...")
                    embeddings = torch.nan_to_num(embeddings, nan=0.0, posinf=1.0, neginf=-1.0)

                # Align to original sequence length
                aligned_embeddings = self.align_embeddings(embeddings, inputs, original_sequence)

                return aligned_embeddings

        except Exception as e:
            print(f"Model inference failed: {e}")
            raise e

    def align_embeddings(self, embeddings, inputs, original_sequence):
        """Align embeddings to original sequence length"""
        try:
            target_length = len(original_sequence)
            current_length = embeddings.shape[0]

            print(f"Aligning embeddings:")
            print(f"   Target length: {target_length}")
            print(f"   Current length: {current_length}")

            if current_length == target_length:
                print("Lengths already match")
                return embeddings

            # Get tokens to understand structure
            tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
            print(f"   Token examples: {tokens[:10]}...{tokens[-5:]}")

            # Find actual amino acid token positions
            aa_positions = []
            valid_aas = set('ACDEFGHIKLMNPQRSTVWYXBZUO')

            for i, token in enumerate(tokens):
                # Clean token
                clean_token = token.replace('▁', '').replace('<', '').replace('>', '').strip()
                if len(clean_token) == 1 and clean_token.upper() in valid_aas:
                    aa_positions.append(i)

            print(f"   Found {len(aa_positions)} amino acid token positions")

            if len(aa_positions) == target_length:
                # Perfect match
                aligned = embeddings[aa_positions]
                print("Aligned using amino acid positions")
            elif len(aa_positions) > target_length:
                # Truncate
                aligned = embeddings[aa_positions[:target_length]]
                print("Truncated to target length")
            else:
                # Heuristic alignment
                if current_length > target_length:
                    # Remove special tokens then uniformly sample
                    start_idx = 1 if tokens[0] in ['<s>', '<cls>', '[CLS]'] else 0
                    end_idx = current_length - 1 if tokens[-1] in ['</s>', '<eos>', '[SEP]'] else current_length

                    available_length = end_idx - start_idx
                    if available_length >= target_length:
                        indices = np.linspace(start_idx, end_idx-1, target_length, dtype=int)
                        aligned = embeddings[indices]
                        print("Aligned using uniform sampling")
                    else:
                        # Direct truncation
                        aligned = embeddings[start_idx:start_idx+target_length]
                        print("Aligned using direct truncation")
                else:
                    # Padding
                    pad_size = target_length - current_length
                    padding = torch.zeros(pad_size, embeddings.shape[1], device=embeddings.device)
                    aligned = torch.cat([embeddings, padding], dim=0)
                    print("Aligned using zero padding")

            print(f"Alignment complete: {aligned.shape}")
            return aligned

        except Exception as e:
            print(f"Alignment failed: {e}")
            # Fallback: simple truncation or padding
            if embeddings.shape[0] > len(original_sequence):
                return embeddings[:len(original_sequence)]
            elif embeddings.shape[0] < len(original_sequence):
                pad_size = len(original_sequence) - embeddings.shape[0]
                padding = torch.zeros(pad_size, embeddings.shape[1], device=embeddings.device)
                return torch.cat([embeddings, padding], dim=0)
            else:
                return embeddings

    def extract_features(self, pdb_path, output_dir="./fixed_ankh_features"):
        """Extract features for a single PDB file"""
        try:
            pdb_name = Path(pdb_path).stem
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            output_path = output_dir / f"{pdb_name}.npy"

            print(f"\n{'='*60}")
            print(f"Processing: {pdb_name}")
            print(f"{'='*60}")

            # Step 1: Extract sequence
            sequence = self.extract_sequence_from_pdb(pdb_path)

            # Step 2: Tokenization
            inputs = self.safe_tokenize(sequence)

            # Step 3: Model inference
            embeddings = self.safe_inference(inputs, sequence)

            # Step 4: Final check
            final_embeddings = embeddings.cpu().numpy().astype(np.float32)

            print(f"Final feature check:")
            print(f"   Shape: {final_embeddings.shape}")
            print(f"   dtype: {final_embeddings.dtype}")
            print(f"   Value range: [{final_embeddings.min():.6f}, {final_embeddings.max():.6f}]")
            print(f"   Mean: {final_embeddings.mean():.6f}")
            print(f"   Std: {final_embeddings.std():.6f}")
            print(f"   NaN count: {np.isnan(final_embeddings).sum()}")
            print(f"   Inf count: {np.isinf(final_embeddings).sum()}")

            # Final safety check
            if np.isnan(final_embeddings).any() or np.isinf(final_embeddings).any():
                print("Residual abnormal values found, performing final cleanup")
                final_embeddings = np.nan_to_num(final_embeddings, nan=0.0, posinf=1.0, neginf=-1.0)

            # Save features
            np.save(output_path, final_embeddings)

            print(f"Features saved: {output_path}")
            print(f"   File size: {output_path.stat().st_size / (1024*1024):.2f} MB")

            return True

        except Exception as e:
            print(f"Feature extraction failed {Path(pdb_path).stem}: {e}")
            import traceback
            traceback.print_exc()
            return False

    def batch_extract(self, pdb_folder, output_dir="./fixed_ankh_features"):
        """Batch extract features"""
        pdb_dir = Path(pdb_folder)
        pdb_files = list(pdb_dir.glob("*.pdb"))

        if not pdb_files:
            print(f"No PDB files found in {pdb_folder}")
            return

        print(f"Starting batch processing of {len(pdb_files)} PDB files")
        print(f"Output directory: {output_dir}")

        success_count = 0
        failed_files = []

        for i, pdb_file in enumerate(pdb_files, 1):
            print(f"\nProgress: {i}/{len(pdb_files)}")

            try:
                if self.extract_features(str(pdb_file), output_dir):
                    success_count += 1
                else:
                    failed_files.append(pdb_file.name)
            except Exception as e:
                print(f"Processing failed: {pdb_file.name} - {e}")
                failed_files.append(pdb_file.name)

        print(f"\n{'='*60}")
        print(f"Batch processing complete")
        print(f"{'='*60}")
        print(f"Success: {success_count}/{len(pdb_files)}")
        print(f"Failed: {len(failed_files)}")

        if failed_files:
            print(f"\nFailed file list:")
            for file in failed_files[:10]:  # Show only first 10
                print(f"  - {file}")
            if len(failed_files) > 10:
                print(f"  ... and {len(failed_files) - 10} more")

def main():
    """Main function"""
    # Configuration
    ANKH_MODEL_PATH = os.environ.get("ANKH_MODEL_PATH", "./ankh-large-model")
    OUTPUT_DIR = "./rna_ankh_features"

    # PDB folder list
    PDB_FOLDERS = [
        "./data/RNA-117_Test",
        "./data/RNA-495_Train",
    ]

    try:
        # Create extractor
        extractor = FixedAnkhExtractor(ANKH_MODEL_PATH)

        # Batch process all folders
        for pdb_folder in PDB_FOLDERS:
            if Path(pdb_folder).exists():
                print(f"\nProcessing folder: {pdb_folder}")
                extractor.batch_extract(pdb_folder, OUTPUT_DIR)
            else:
                print(f"Folder does not exist: {pdb_folder}")

        print(f"\nAll processing complete!")
        print(f"Please use the feature verification tool to validate the generated feature files")

    except Exception as e:
        print(f"Program execution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
