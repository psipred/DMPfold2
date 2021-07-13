# DMPfold2

DMPfold2 is a fast and accurate method for protein structure prediction.

If you use DMPfold2, please cite the paper: Deep learning-based prediction of protein structure using learned representations of multiple sequence alignments, S M Kandathil, J G Greener, A M Lau, D T Jones, bioRxiv (2021).

## Installation

DMPfold2 is easier to install than DMPfold1, which had many more dependencies.

1. Python 3.6 or later is required.

2. Install [PyTorch](https://pytorch.org) as appropriate for your system.

3. Run `pip install dmpfold`. The repository takes up ~150 MB of disk space.

## Usage

Run `dmpfold -h` to see a help message.
To run DMPfold2 you will need a sequence alignment in `aln` format: one sequence per line with the ungapped target sequence as the first line.
Sequence alignments can be obtained from a target sequence in a number of ways, for example by running `hhblits` on the Uniclust database.

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

### CASP14 version

If for some reason you need the CASP14 version of the developing DMPfold2, run `git checkout casp14` on this repository and find instructions in the readme file.
This version used three approaches to generate models from constraints - CNS, XPLOR-NIH and a PyTorch-based molecular dynamics approach - but is less accurate, slower and harder to install than the current end-to-end approach.
