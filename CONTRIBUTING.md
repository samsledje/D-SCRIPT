# Contributing

- Install conda environment
```
conda env create -f environment.yml
conda activate dscript
```
- Download required development packages

```
pip install pre-commit isort flake8 black pytest coverage[toml]
```

- Install pre-commit
```
pre-commit install
```
