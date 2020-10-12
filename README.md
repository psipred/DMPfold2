# DMPfold2

Development code for DMPfold2 distribution (currently private).

## Installation

We have attempted to make DMPfold2 easier to install than DMPfold1, which had many more dependencies.

1. Python 3.6 or later is required.
2. Install [PyTorch](https://pytorch.org) as appropriate for your system. It is not necessary to have a GPU, though it is recommended if using the force-directed folding protocol (see below).
3. Install [MODELLER](https://salilab.org/modeller), which is usually as simple as running `conda install -c salilab modeller` and modifying the license key.
4. Install DMPfold2:
```bash
git clone https://github.com/psipred/DMPfold2 # Enter GitHub username and password
cd DMPfold2
pip install -e .
```
[This will become `pip install dmpfold` on registering in PyPI.]
5. Copy or symlink trained NN models to relevant place:
```bash
mkdir -p dmpfold/nn/multigram dmpfold/nn/multigram-iter # From DMPfold2 directory
for i in {1..4}; do
    ln -s path/to/multigram/FINAL_fullmap_distcov_model$i.pt dmpfold/nn/multigram/FINAL_fullmap_distcov_model$i.pt
done
for i in {1..3}; do
    ln -s path/to/multigram-iter/FINAL_fullmap_distcov_model$i.pt dmpfold/nn/multigram-iter/FINAL_fullmap_distcov_model$i.pt
done
```
[This step will disappear once we work out the best way to upload the 120 MB network files to GitHub, which has a 100 MB file size limit.]

This will automatically install NumPy, SciPy and PeptideBuilder if required, and will put the executable `dmpfold` on the path.

The above steps are sufficient to predict distograms with DMPfold2 from a sequence alignment.
To generate models, you will need to set up at least one of the following 3 approaches.
[See the DMPfold2 paper for a comparison of the approaches.]

### CNS

[CNS install instructions.]

It is assumed that the `cns_solve_env.sh` setup script has been run before `dmpfold` is called, and hence the `cns` command is available.
You might want to put it in your bash profile.

### Xplor-NIH

[Xplor-NIH install instructions.]

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
where `out_dir` is the output directory to write results to (which will be created).

### Getting sequence alignments

Sequence alignments can be obtained from a target sequence in a number of ways, for example by running `hhblits` on the Uniclust database.

[Example script and conversion to aln.]

### Predicting distances

Residue-residue distances, hydrogen bonds and dihedral angles can be predicted using the Python interface:
```python
from dmpfold import aln_to_predictions, write_predictions
output = aln_to_predictions(aln_filepath) # Shape (1, 104, n_res, n_res)
write_predictions(output, "out")
```
You can also run structure prediction from within Python, for example:
```python
from dmpfold import aln_to_model_fdf
aln_to_model_fdf("prot.aln", "out_dir") # Equivalent to `dmpfold -i prot.aln -o out_dir -p fdf`
```
