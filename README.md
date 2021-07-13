# DMPfold2

This is the CASP14 version of DMPfold2, not the final version.
If you use DMPfold2, please cite the paper.

## Installation

We have attempted to make DMPfold2 easier to install than DMPfold1, which had many more dependencies.

1. Python 3.6 or later is required.

2. Install [PyTorch](https://pytorch.org) as appropriate for your system. It is not necessary to have a GPU, though it is recommended if using the force-directed folding protocol (see below).

3. Install [MODELLER](https://salilab.org/modeller), which is usually as simple as running `conda install -c salilab modeller` and modifying the license key.

4. Install DMPfold2 by downloading or cloning the repository from GitHub then running `pip install -e .` from the `DMPfold2` directory. This will automatically install NumPy, SciPy and PeptideBuilder if required, and will put the executable `dmpfold` on the path.

5. Download the trained NN models and move them to the relevant place:
```bash
wget http://bioinfadmin.cs.ucl.ac.uk/downloads/dmpfold2/casp14_trained_models.tar.gz
tar -xvf casp14_trained_models.tar.gz
mv nn path/to/DMPfold2/dmpfold
```

The above steps are sufficient to predict distograms with DMPfold2 from a sequence alignment.
To generate models, you will need to set up at least one of the following 3 approaches.

### CNS

[CNS installation instructions.]

It is assumed that the `cns_solve_env.sh` setup script has been run before `dmpfold` is called, and hence the `cns` command is available.
You might want to put it in your bash profile.

### Xplor-NIH

[Xplor-NIH installation instructions.]

When running DMPfold2, the `-x` flag should point to the Xplor-NIH `bin` directory and `-n` can be used to set the number of CPUs for model generation (default 4).

### Force-directed folding

No extra setup is required.
However, it is strongly recommended that you have a GPU available as folding takes considerably longer on the CPU.

## Usage

Run `dmpfold -h` to see a help message.
To run DMPfold2 you will need the sequence alignment in `aln` format: one sequence per line with the ungapped target sequence as the first line.
To run, choose a protocol:
```bash
dmpfold -i prot.aln -o out_dir -p cns # CNS protocol, assumes the `cns` command is available

dmpfold -i prot.aln -o out_dir -p xplor -x ~/xplor-nih-3.1/bin # Xplor-NIH protocol

dmpfold -i prot.aln -o out_dir -p fdf # Force-directed folding protocol
```
`out_dir` is the output directory to write results to, which will be created.

### Getting sequence alignments

Sequence alignments can be obtained from a target sequence in a number of ways, for example by running `hhblits` on the Uniclust database.

### Predicting distances

Residue-residue distances, hydrogen bonds and dihedral angles can be predicted using the Python interface:
```python
from dmpfold import aln_to_predictions, write_predictions
output = aln_to_predictions(aln_filepath) # Shape (1, 104, n_res, n_res)
write_predictions(output, "out") # Writes out.dist, out.hb, out.phi and out.psi
```
You can also run structure prediction from within Python, for example:
```python
from dmpfold import aln_to_model_fdf
aln_to_model_fdf("prot.aln", "out_dir") # Equivalent to `dmpfold -i prot.aln -o out_dir -p fdf`
```
