# D-SCRIPT - AI Assistant Development Guide

## Project Overview

**D-SCRIPT** (Deep Learning for Structure-Aware Protein-Protein Interaction Prediction) is a deep learning method for predicting physical interactions between proteins using only their sequences. The project uses structure-aware design to enhance interpretability and cross-species generalizability.

**Current Version**: 0.3.1
**License**: MIT
**Python Requirements**: >=3.10
**Primary Frameworks**: PyTorch (>=1.13), NumPy, Pandas, SciKit-Learn

### Key Features
- Protein-protein interaction (PPI) prediction from sequences
- Language model-based protein embeddings (Bepler+Berger)
- Structure-aware contact map predictions
- Multi-GPU support for inference with blocked memory-efficient processing
- Pre-trained models available via HuggingFace
- Supports Topsy-Turvy (network information) and TT3D (structure information)

### Published Research
- Original D-SCRIPT: Cell Systems (2021) DOI: 10.1016/j.cels.2021.08.010
- Topsy-Turvy: Bioinformatics (2022)
- TT3D: Bioinformatics (2023)
- BMPI: Bioinformatics (2024) DOI: 10.1093/bioinformatics/btaf564

---

## Repository Structure

```
D-SCRIPT/
├── dscript/                    # Main Python package
│   ├── __init__.py            # Package initialization (v0.3.1)
│   ├── __main__.py            # CLI entry point with argparse subcommands
│   ├── alphabets.py           # Protein sequence alphabets
│   ├── fasta.py               # FASTA file parsing (uses biotite)
│   ├── foldseek.py            # FoldSeek 3Di structure sequence support
│   ├── glider.py              # Data loading utilities
│   ├── language_model.py      # Bepler+Berger language model
│   ├── pretrained.py          # Pre-trained model loading
│   ├── utils.py               # Logging, device parsing, data structures
│   ├── loading.py             # Parallel HDF5 loading
│   ├── load_worker.py         # Worker for parallel loading
│   ├── commands/              # CLI command implementations
│   │   ├── embed.py           # Generate embeddings from sequences
│   │   ├── train.py           # Model training
│   │   ├── evaluate.py        # Model evaluation metrics
│   │   ├── predict_serial.py  # Serial prediction (legacy)
│   │   ├── predict_block.py   # Blocked multi-GPU prediction (default)
│   │   ├── predict_bipartite.py # Cross-species prediction
│   │   ├── extract_3di.py     # Extract 3Di sequences
│   │   ├── par_worker.py      # Parallel prediction worker
│   │   └── par_writer.py      # Parallel prediction writer
│   ├── models/                # Neural network architectures
│   │   ├── embedding.py       # Projection layers (FullyConnectedEmbed, SkipLSTM)
│   │   ├── contact.py         # Contact map CNN
│   │   └── interaction.py     # Main interaction model (ModelInteraction, DSCRIPTModel)
│   └── tests/                 # Unit tests (pytest)
│       ├── test_fasta.py
│       ├── test_alphabets.py
│       ├── test_models_*.py
│       ├── test_commands.py
│       └── ...
├── data/                      # Sample data and test files
│   ├── seqs/                  # FASTA sequences
│   ├── pairs/                 # TSV pair files
│   └── *.fa                   # FoldSeek sequences
├── docs/                      # Sphinx documentation
│   └── source/
├── bash_files/                # Training/testing bash scripts
├── scripts/                   # Utility scripts
│   ├── bmpi_bench/           # BMPI benchmarking
│   └── push_huggingface.py   # HuggingFace model upload
├── notebooks/                 # Jupyter notebooks
├── pyproject.toml            # Package configuration
├── environment.yml           # Conda environment
├── .pre-commit-config.yaml   # Pre-commit hooks
├── .github/workflows/        # CI/CD pipelines
└── CHANGELOG.md             # Version history
```

---

## Architecture & Core Components

### Neural Network Architecture

The D-SCRIPT model follows a three-stage pipeline:

1. **Embedding**: Projects protein language model embeddings to lower dimensions
   - `FullyConnectedEmbed`: Fully-connected projection with dropout
   - `SkipLSTM`: LSTM-based projection (for language model)

2. **Contact Prediction**: Predicts inter-protein contact maps
   - `ContactCNN`: CNN operating on outer product of embeddings
   - Outputs contact probability map (N × M)

3. **Interaction Prediction**: Aggregates contact maps to interaction probability
   - `ModelInteraction`: Main model combining embedding + contact
   - Weighted pooling with learnable parameters (θ, λ, γ)
   - Logistic activation for final prediction

### Model Classes

**`ModelInteraction`** (dscript/models/interaction.py:51)
- Core model class combining all components
- Key methods:
  - `embed(x)`: Project embeddings
  - `cpred(z0, z1, ...)`: Predict contact map
  - `predict(z0, z1, ...)`: Predict interaction probability
  - `map_predict(z0, z1, ...)`: Return both contact map and probability

**`DSCRIPTModel`** (dscript/models/interaction.py:265)
- HuggingFace-compatible wrapper
- Inherits from `ModelInteraction` and `PyTorchModelHubMixin`
- Used for saving/loading models to HuggingFace Hub

### Pre-trained Models

Pre-trained models are managed via `dscript/pretrained.py`:

- `human_v1`: Original D-SCRIPT (Cell Systems 2021)
- `human_v2`: Topsy-Turvy (default, recommended)
- `human_tt3d`: TT3D with FoldSeek 3Di sequences
- `lm_v1`: Bepler & Berger language model

Models can be loaded from:
1. HuggingFace Hub (e.g., `samsl/topsy_turvy_human_v1`)
2. Local disk via `get_pretrained(version)`
3. Direct state dict download from MIT server

### Data Formats

**Input Formats**:
- **FASTA** (.fasta): Protein sequences (parsed with biotite)
- **TSV** (.tsv): Protein pairs (tab-separated, no header)
  - Format: `protein1\tprotein2\t[optional_label]`
- **HDF5** (.h5): Embeddings storage
  - Keys: protein identifiers
  - Values: embedding tensors

**Output Formats**:
- Predictions: TSV with scores
- Embeddings: HDF5 files
- Trained models: PyTorch state dicts (.pt) or HuggingFace format

---

## CLI Commands

The package provides a unified CLI via `dscript` command:

### Main Commands

1. **`dscript embed`** - Generate embeddings
   ```bash
   dscript embed --seqs data/seqs/ecoli.fasta --outfile ecoli_embed.h5
   ```

2. **`dscript predict`** - Predict interactions (blocked mode, default)
   ```bash
   dscript predict --pairs data/pairs/ecoli.tsv \
                   --embeddings ecoli_embed.h5 \
                   --model samsl/topsy_turvy_human_v1 \
                   --outfile predictions.tsv \
                   --blocks 16 -d 0
   ```

3. **`dscript predict_serial`** - Serial prediction (legacy)

4. **`dscript predict_bipartite`** - Cross-species prediction

5. **`dscript train`** - Train new model
   ```bash
   dscript train --train train.tsv --test test.tsv \
                 --embedding embeddings.h5 \
                 --output output_dir --save-prefix model
   ```

6. **`dscript evaluate`** - Evaluate model performance

7. **`dscript extract-3di`** - Extract FoldSeek 3Di sequences

### CLI Architecture

The CLI is implemented using argparse subcommands (dscript/__main__.py):
- Each command is a module in `dscript/commands/`
- Each module provides `add_args(parser)` and `main(args)` functions
- Type hints use union types for argument classes

---

## Development Workflows

### Environment Setup

**Using Conda** (recommended):
```bash
conda env create -f environment.yml
conda activate dscript
```

**Using pip**:
```bash
pip install -e ".[dev]"  # Editable install with dev dependencies
```

### Code Quality Tools

**Ruff** - Linting and formatting (configured in pyproject.toml:66-87)
- Line length: 90 characters
- Target: Python 3.10+
- Enabled rules: pycodestyle (E/W), pyflakes (F), isort (I), pyupgrade (UP)
- Ignored: E501 (line too long)
- Quote style: double quotes
- Indentation: spaces

**Pre-commit Hooks** (.pre-commit-config.yaml):
1. Check merge conflicts
2. Check YAML syntax
3. Fix end-of-file
4. Trim trailing whitespace
5. Ruff check with auto-fix
6. Ruff format

**Setting up pre-commit**:
```bash
pip install pre-commit
pre-commit install
```

### Running Linters

```bash
# Check code
ruff check . --statistics

# Format code
ruff format .

# Run pre-commit on all files
pre-commit run --all-files
```

---

## Testing

### Test Framework

- **Framework**: pytest
- **Location**: `dscript/tests/`
- **Coverage**: pytest-cov

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=dscript --cov-report=xml --cov-report=term-missing

# Run specific test file
pytest dscript/tests/test_fasta.py

# Run specific test
pytest dscript/tests/test_models_interaction.py::test_model_forward
```

### Test Configuration

Configuration in pyproject.toml:92-101:
- Test file pattern: `test_*.py`
- Test path: `dscript/tests`
- Warnings filtered: UserWarning, DeprecationWarning, PendingDeprecationWarning

### Test Coverage

Current test files cover:
- `test_fasta.py`: FASTA parsing
- `test_alphabets.py`: Protein alphabets
- `test_models_embedding.py`: Embedding layers
- `test_models_contact.py`: Contact prediction
- `test_models_interaction.py`: Full interaction model
- `test_pretrained.py`: Model loading
- `test_language_model.py`: Language model
- `test_foldseek.py`: 3Di sequence handling
- `test_commands.py`: CLI commands

---

## CI/CD Pipelines

### GitHub Actions Workflows

**1. pytest** (.github/workflows/autorun-tests.yml)
- Triggers: Push/PR to main branch
- Steps:
  1. Setup Python 3.10
  2. Install dependencies with dev/test extras
  3. Run ruff check and format
  4. Run pytest with coverage
  5. Generate coverage.xml

**2. docs-build** (.github/workflows/docs-build.yml)
- Builds Sphinx documentation
- Publishes to ReadTheDocs

**3. python-publish** (.github/workflows/python-publish.yml)
- Publishes package to PyPI
- Triggered on release tags

---

## Key Coding Conventions

### Code Style

1. **Line Length**: 90 characters (pyproject.toml:67)
2. **Quotes**: Double quotes for strings
3. **Imports**: Sorted with isort, grouped by:
   - Standard library
   - Third-party packages
   - Local imports (dscript.*)
4. **Type Hints**: Use type hints for function signatures where applicable
5. **Docstrings**: Use reStructuredText format for Sphinx

### Naming Conventions

- **Variables**: snake_case (e.g., `state_dict_path`)
- **Functions**: snake_case (e.g., `get_pretrained`)
- **Classes**: PascalCase (e.g., `ModelInteraction`, `ContactCNN`)
- **Constants**: UPPER_CASE (e.g., `VALID_MODELS`, `ROOT_URL`)
- **Private**: Leading underscore (e.g., `self._internal_method`)

### Model Design Patterns

1. **Module Composition**: Models compose smaller modules
   ```python
   embedding = FullyConnectedEmbed(...)
   contact = ContactCNN(...)
   model = ModelInteraction(embedding, contact, ...)
   ```

2. **Device Handling**: Pass `use_cuda` flag to models
   ```python
   model = DSCRIPTModel(..., use_cuda=True)
   ```

3. **Parameter Clamping**: Use `.clip()` methods to constrain parameters
   ```python
   def clip(self):
       self.theta.data.clamp_(min=0, max=1)
   ```

### Logging

The project uses **loguru** for logging (migrated in v0.3.0):

```python
from dscript.utils import log, setup_logger

# Legacy compatibility wrapper
log("Message", file=log_file, print_also=True)

# Direct loguru usage
from loguru import logger
logger.info("Message")
```

### Error Handling

1. **Validate inputs**: Check file existence, data shapes, CUDA availability
2. **Informative errors**: Provide context in error messages
3. **Graceful degradation**: Handle download failures with retries
4. **Exit codes**: Use `sys.exit(1)` for fatal errors

Example (dscript/utils.py:96-130):
```python
def parse_device(device_arg, logFile):
    if use_cuda and not torch.cuda.is_available():
        log("CUDA not available but GPU requested...", ...)
        sys.exit(1)
```

---

## Working with the Codebase

### Adding a New Model

1. Create model class in `dscript/models/`
2. Inherit from appropriate base (nn.Module, PyTorchModelHubMixin)
3. Implement `forward()`, `clip()` if needed
4. Add to pretrained.py if pre-trained version exists
5. Write unit tests in `dscript/tests/test_models_*.py`

### Adding a New Command

1. Create file in `dscript/commands/new_command.py`
2. Implement:
   ```python
   def add_args(parser):
       parser.add_argument(...)

   def main(args):
       # Implementation
   ```
3. Add to `__main__.py` modules dict
4. Create TypedDict for arguments (optional)
5. Write tests in `dscript/tests/test_commands.py`

### Modifying Data Loading

Data loading is parallelized using `LoadingPool` (dscript/loading.py):
- HDF5 files loaded via `load_hdf5_parallel()`
- Multiprocessing pool for parallel access
- Used extensively in prediction commands

When modifying:
- Test with various `n_jobs` settings
- Ensure thread-safety for HDF5 access
- Validate key existence before loading

### Working with Embeddings

Embeddings are stored in HDF5 format with structure:
```
embeddings.h5
├── protein1 -> [N × d] array
├── protein2 -> [N × d] array
└── ...
```

Key considerations:
- Protein names (keys) must match between pairs TSV and embeddings
- Sequences can have variable length (N)
- Embedding dimension (d) must match model expectations
- Use `glider.py` utilities for reading/writing

---

## Common Tasks

### Running the Full Pipeline

```bash
# 1. Generate embeddings
dscript embed --seqs proteins.fasta --outfile embeddings.h5

# 2. Make predictions
dscript predict --pairs pairs.tsv \
                --embeddings embeddings.h5 \
                --model samsl/topsy_turvy_human_v1 \
                --outfile predictions.tsv

# 3. Evaluate (if labels available)
dscript evaluate --pairs test_pairs.tsv \
                 --embeddings embeddings.h5 \
                 --model samsl/topsy_turvy_human_v1 \
                 --outfile metrics.json
```

### Training a Custom Model

```bash
dscript train --train train.tsv \
              --test test.tsv \
              --embedding embeddings.h5 \
              --output output_dir \
              --save-prefix my_model \
              --device 0 \
              --batch-size 32 \
              --num-epochs 100
```

### Using Multi-GPU Prediction

```bash
# Use all GPUs with 16 blocks
dscript predict --pairs large_pairs.tsv \
                --embeddings embeddings.h5 \
                --model samsl/topsy_turvy_human_v1 \
                --blocks 16 \
                -d all \
                --outfile predictions.tsv
```

### Loading Pre-trained Models in Code

```python
from dscript.pretrained import get_pretrained
from dscript.models.interaction import DSCRIPTModel
from huggingface_hub import hf_hub_download

# Option 1: Load from built-in
model = get_pretrained("human_v2")

# Option 2: Load from HuggingFace
model = DSCRIPTModel.from_pretrained("samsl/topsy_turvy_human_v1")

# Option 3: Load from local file
model = DSCRIPTModel.from_pretrained("/path/to/model/")
```

### Adding Tests

```python
# dscript/tests/test_new_feature.py
import pytest
import torch
from dscript.models.interaction import ModelInteraction

def test_model_forward():
    """Test model forward pass"""
    model = ModelInteraction(...)
    z0 = torch.randn(1, 100, 100)
    z1 = torch.randn(1, 100, 100)

    output = model(z0, z1)

    assert output.shape == torch.Size([1])
    assert 0 <= output.item() <= 1
```

---

## Important Notes for AI Assistants

### When Making Changes

1. **Always read files before modifying**: Don't propose changes to code you haven't seen
2. **Run tests after changes**: Use `pytest` to verify functionality
3. **Check code style**: Run `ruff check` and `ruff format` before committing
4. **Update documentation**: If changing APIs, update docstrings and docs/
5. **Follow version history**: Check CHANGELOG.md for context on previous changes

### Security Considerations

- **Input validation**: Always validate file paths, URLs (dscript/pretrained.py:107-111)
- **Device validation**: Check CUDA availability before GPU operations
- **URL schemes**: Only allow http/https for downloads
- **No arbitrary code execution**: Don't use eval() or exec()

### Performance Considerations

- **Memory efficiency**: Use blocked loading for large datasets (predict_block.py)
- **GPU utilization**: Support multi-GPU with `device=-1`
- **Parallel processing**: Use multiprocessing for HDF5 loading
- **Sparse loading**: Only load required embeddings when possible

### Common Pitfalls

1. **CUDA availability**: Always check `torch.cuda.is_available()` before GPU ops
2. **Model eval mode**: Set `model.eval()` before inference (disables dropout)
3. **Protein key matching**: Ensure FASTA headers match TSV pair identifiers
4. **Embedding dimensions**: Verify embedding dimension matches model expectations
5. **File handles**: Close log files and HDF5 files properly

### Useful References

- **Documentation**: https://d-script.readthedocs.io/
- **PyPI Package**: https://pypi.org/project/dscript/
- **GitHub**: https://github.com/samsledje/D-SCRIPT
- **HuggingFace Demo**: https://huggingface.co/spaces/samsl/D-SCRIPT
- **Paper**: https://doi.org/10.1016/j.cels.2021.08.010

---

## Git Workflow

### Branching Strategy

- **main**: Stable release branch
- **feature branches**: `feature/description` or `claude/session-id`
- **bug fixes**: `fix/description`

### Commit Messages

Follow conventional commits style:
- `feat: Add new model architecture`
- `fix: Resolve CUDA device error`
- `docs: Update API documentation`
- `test: Add unit tests for contact model`
- `refactor: Simplify embedding loading`
- `chore: Update dependencies`

### Before Pushing

```bash
# Run pre-commit hooks
pre-commit run --all-files

# Run tests
pytest

# Check for common issues
ruff check .
```

---

## Package Distribution

### PyPI Release Process

1. Update version in `dscript/__init__.py`
2. Update CHANGELOG.md
3. Commit changes
4. Create git tag: `git tag v0.3.1`
5. Push tag: `git push origin v0.3.1`
6. GitHub Action automatically publishes to PyPI

### HuggingFace Model Upload

Use `scripts/push_huggingface.py` to upload trained models to HuggingFace Hub.

---

## Project Maintenance

### Dependencies

Core dependencies (pyproject.toml:15-29):
- **torch** (>=1.13): Deep learning framework
- **biotite** (==1.2.0): FASTA parsing (replaced BioPython in v0.3.1)
- **numpy, scipy, pandas**: Numerical computing
- **scikit-learn**: Metrics and utilities
- **h5py**: HDF5 file handling
- **huggingface_hub**: Model hosting
- **loguru**: Logging (migrated in v0.3.0)
- **tqdm**: Progress bars

### Known Issues & TODOs

From CHANGELOG.md:
- Expand test suite to maximize coverage
- Continue improving documentation

### Version History Highlights

- **v0.3.1**: Replaced BioPython with biotite, removed local foldseek requirement
- **v0.3.0**: Major modernization - BMPI support, loguru migration, Ruff adoption
- **v0.2.0**: Topsy-Turvy integration, parallel HDF5 loading
- **v0.1.8**: Training bug fixes for paper replication

---

*This guide is current as of v0.3.1 (2025). For the latest updates, refer to CHANGELOG.md and the official documentation.*
