# NucSeqOptimizer

## Quick start (local dev on macOS with MPS)

This repo includes Chai-1 as a vendored submodule under `chai_lab/` for tests and utilities. You can run all unit tests and the example folding pipeline locally on an Apple Silicon Mac without CUDA using PyTorch (MPS). The full Chai-1 model is optimized for Linux + CUDA; macOS runs are for testing and small demos.

### Setup (pyenv + venv, no conda)

```bash
# Prereqs
brew install pyenv kalign
pyenv install -s 3.10.14
pyenv local 3.10.14
python3 -m venv .venv310
source .venv310/bin/activate
pip install --upgrade pip setuptools wheel

# Project deps
pip install -e chai_lab  # installs chai_lab and its deps
pip install pyyaml  # used by scripts/make_fasta_and_run.py

# PyTorch (CPU/MPS build)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

Verify:

```bash
python - <<'PY'
import torch
print('torch:', torch.__version__)
print('mps:', torch.backends.mps.is_available())
PY
```

### Run tests

```bash
source .venv310/bin/activate
pytest -q chai_lab/tests -k 'not colabfold_msas'
```

### Generate FASTA and run Chai-1 locally

Use `scripts/make_fasta_and_run.py` to convert a YAML spec to FASTA and run the Chai-1 CLI if available. The script now:

- Writes headers compatible with Chai (`>protein|name=A`, etc.)
- Picks a device automatically (MPS on macOS, else CUDA if available, else CPU) unless overridden via `--device`
- Creates a timestamped subdirectory under the output dir for each run
- Supports lighter inference runs via flags

Example (MPS, resource-friendly):

```bash
source .venv310/bin/activate
python scripts/make_fasta_and_run.py examples/test_input.yaml outputs \
  --fasta-name input.fasta \
  --no-esm \
  --device mps \
  --trunk-recycles 1 \
  --diffn-timesteps 50 \
  --diffn-samples 1
```

This will produce a CIF at `outputs/run_<timestamp>/pred.model_idx_0.cif`.

Flags:

- `--device {mps|cuda:0|cpu}`: override device
- `--no-esm`: disable ESM embedding
- `--trunk-recycles N`: trunk recycles (default 3)
- `--diffn-timesteps N`: diffusion steps (default 200)
- `--diffn-samples N`: number of diffusion samples (default 5)
- `--use-msa-server` and `--msa-server-url URL`: generate MSAs via the ColabFold server (networked)

### Remote Linux GPU run

If you have a Linux GPU host accessible over SSH, use `scripts/run_remote_chai.py` to run remotely and fetch outputs. See the script `--help` for options. Example:

```bash
python scripts/run_remote_chai.py examples/test_input.yaml outputs \
  --remote user@linux-gpu-host \
  --use-msa-server \
  --install-chai
```

## Notes

- Kalign is required for template alignment and is auto-detected from PATH.
- Weights are downloaded on first run to `chai_lab/downloads` by default; override via `CHAI_DOWNLOADS_DIR=/path`.
- The vendored `chai_lab` tests are pinned to Python 3.10 and pass on macOS MPS when skipping ColabFold network tests.

## License

Apache-2.0
