# D-SCRIPT
[![PyPI](https://img.shields.io/pypi/v/dscript)](https://pypi.org/project/dscript/)
[![DOI](https://zenodo.org/badge/308463847.svg)](https://zenodo.org/badge/latestdoi/308463847)
[![License](https://img.shields.io/github/license/samsledje/D-SCRIPT)](https://github.com/samsledje/D-SCRIPT/blob/main/LICENSE)
[![Pytest](https://github.com/samsledje/D-SCRIPT/actions/workflows/autorun-tests.yml/badge.svg)](https://github.com/samsledje/D-SCRIPT/actions/workflows/autorun-tests.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

![D-SCRIPT Architecture](docs/source/img/dscript_architecture.png)

D-SCRIPT is a deep learning method for predicting a physical interaction between two proteins given just their sequences. It generalizes well to new species and is robust to limitations in training data size. Its design reflects the intuition that for two proteins to physically interact, a subset of amino acids from each protein should be in contact with the other. The intermediate stages of D-SCRIPT directly implement this intuition, with the penultimate stage in D-SCRIPT being a rough estimate of the inter-protein contact map of the protein dimer. This structurally-motivated design enhances the interpretability of the results and, since structure is more conserved evolutionarily than sequence, improves generalizability across species.

You can now make predictions with D-SCRIPT via the interface on [HuggingFace](https://huggingface.co/spaces/samsl/D-SCRIPT)!

## Installation

```bash
pip install dscript
```

## Usage

Protein sequences need to first be embedded using the [Bepler+Berger](https://doi.org/10.48550/arXiv.1902.08661) protein language model; this requires a `.fasta` file as input. Everything before the first space will be used as the key.

```bash
dscript embed --seqs [sequences] --outfile [embedding file]

#Example
dscript embed --seqs data/seqs/ecoli.fasta --outfile ecoli_embed.h5

```

Candidate pairs should be in tab-separated (.tsv) format with no header, and columns for [protein key 1], [protein key 2]. Optionally, a third column with [label] can be provided, so predictions can be made using training or test data files (but the label will not affect the predictions only the first two columns will be read).

While [pre-trained model files](https://d-script.readthedocs.io/en/main/data.html#trained-models) can be downloaded directly, we recommend instead passing the name of a pre-trained model that will be automatically downloaded from HuggingFace. Available models include:

- samsl/dscript_human_v1
- samsl/topsy_turvy_human_v1 (**recommended**, default)
- samsl/tt3d_human_v1

```bash
dscript predict --pairs [input data] --embeddings [embedding file] --model [model file] --outfile [predictions file]

#Example
dscript predict --pairs data/pairs/ecoli_toy.tsv --embeddings ecoli_embed.h5 --outfile ecoli_toy_predict
```

For inference, proteins can be divided into blocks to reduce memory usage for embeddings using `--blocks`. By default, the CPU is used; a GPU to use can be specified with `-d`, followed by the index of a GPU or `all` for all available GPUs.
```bash
#Example with 16 blocks, using (using 3/16th the maximum embedding memory), and a GPU
dscript predict --pairs data/pairs/ecoli_test.tsv --embeddings ecoli_embed.h5 --outfile ecoli_test_predict --blocks 16 -d 0
``` 

For more information on prediction modes, such as all-pair and bipartite predictions, see our [complete documentation](https://d-script.readthedocs.io/en/main/usage.html)

## References
 - The original D-SCRIPT model is described in the paper “[D-SCRIPT translates genome to phenome with sequence-based, structure-aware, genome-scale predictions of protein-protein interactions](https://doi.org/10.1016/j.cels.2021.08.010).”
 - We have updated D-SCRIPT to incorporate network information ([Topsy Turvy](https://academic.oup.com/bioinformatics/article/38/Supplement_1/i264/6617505)) and structure information ([TT3D](https://academic.oup.com/bioinformatics/article/39/11/btad663/7332153))
 - The addition of Blocked, Multi-GPU Parallel Inference to D-SCRIPT is described in the application note “[Memory-Efficient, Accelerated Protein Interaction inference with Blocked, Multi-GPU D-SCRIPT](https://doi.org/10.1093/bioinformatics/btaf564).” 
 - [Documentation](https://d-script.readthedocs.io/en/main/)
