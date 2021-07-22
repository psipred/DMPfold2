# DMPfold2

[![Build status](https://github.com/psipred/DMPfold2/workflows/CI/badge.svg)](https://github.com/psipred/DMPfold2/actions)

DMPfold2 is a fast and accurate method for protein structure prediction.
It uses learned representations of multiple sequence alignments and end-to-end model generation to quickly generate models from alignments.

If you use DMPfold2, please cite the paper: Ultrafast end-to-end protein structure prediction enables high-throughput exploration of uncharacterised proteins, S M Kandathil, J G Greener, A M Lau, D T Jones, bioRxiv (2021) - [link](https://www.biorxiv.org/content/10.1101/2020.11.27.401232).

Protein structures predicted from the BFD and Pfam are available via the UCL Research Data Repository (doi: 10.5522/04/14979990) - [link](https://rdr.ucl.ac.uk/articles/dataset/Protein_structures_predicted_using_DMPfold2/14979990).

## Installation

DMPfold2 is easier to install than DMPfold1, which had many more dependencies.

1. Python 3.6 or later is required.

2. Install [PyTorch](https://pytorch.org) as appropriate for your system. A GPU is not required but gives some speedup to longer runs.

3. Run `pip install dmpfold`, which adds the `dmpfold` executable to the path. The first time you run a prediction the trained model files (~140 MB) will be downloaded to the package directory, which requires an internet connection.

## Usage

Run `dmpfold -h` to see a help message.
To run DMPfold2 you will need a sequence alignment in `aln` format: one sequence per line with the ungapped target sequence as the first line (example [here](https://github.com/psipred/DMPfold2/tree/master/dmpfold/example/PF10963.aln)).
Lines starting with `>` are ignored.
Sequence alignments can be obtained from a target sequence in a number of ways, for example by running `hhblits` on the Uniclust database.
DMPfold2 prints a PDB format file to stdout, including the confidence as a remark.

Default mode (10 iteration cycles + 100 steps geometry minimization on cpu device):
```bash
dmpfold -i input.aln > fold.pdb
```

Default mode on cuda device 0:
```bash
dmpfold -i input.aln -d cuda:0 > fold.pdb
```

Fastest mode (no iteration or refinement):
```bash
dmpfold -i input.aln -n 0 -m 0 > fold.pdb
```

30 iteration cycles + 200 steps geometry minimization:
```bash
dmpfold -i input.aln -n 30 -m 200 > fold.pdb
```

If you already have a model (only CA atoms are used) e.g. from HHsearch/MODELLER
(30 iteration cycles + 200 minimization steps + template seed structure):
```bash
dmpfold -i input.aln -n 30 -m 200 -t template.pdb > fold.pdb
```

Ridiculous long run taking hours (100000 iterations + 1000 minimization steps):
```bash
dmpfold -i input.aln -n 100000 -m 1000 > fold.pdb
```

### Python module

DMPfold2 can also be used within Python, allowing you to use it as part of other Python scripts.
For example:
```python
from dmpfold import aln_to_coords

# Default options
coords, confs = aln_to_coords("input.aln")

# Change options
coords, confs = aln_to_coords("input.aln", device="cuda", template="template.pdb", iterations=30, minsteps=200)
```
`coords` is a PyTorch tensor with shape `(nres, 5, 3)` where the first axis is the residue index, the second is the atom (N, CA, C, O, CB) and the third is the coordinates in Angstrom.
`confs` is a PyTorch tensor corresponding to the predicted confidence for each residue.

### CASP14 version

If for some reason you need the CASP14 version of the developing DMPfold2, run `git checkout casp14` on this repository and find instructions in the readme file.
This version used three approaches to generate models from constraints - CNS, XPLOR-NIH and a PyTorch-based molecular dynamics approach - but is less accurate, slower and harder to install than the current end-to-end approach.
