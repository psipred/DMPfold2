# DMPfold2

Development code for DMPfold2 distribution (currently private).

## Installation

We have attempted to make DMPfold2 easier to install than DMPfold1, which had many more dependencies.

1. Python3.6 or later is required.
2. Install [PyTorch](https://pytorch.org) as appropriate for your system. It is not necessary to have a GPU, though it is recommended if using the force-directed folding protocol (see below).
3. Install [MODELLER](https://salilab.org/modeller), which is usually as simple as running `conda install -c salilab modeller` and modifying the license key.
4. To install DMPfold2, run:
```bash
git clone https://github.com/psipred/DMPfold2
cd DMPfold2
pip install -e .
```
[This will become `pip install dmpfold` on registering in PyPI.]
This will automatically install NumPy, SciPy and PeptideBuilder if required and will put the executable `dmpfold` on the path.

The above steps are sufficient to predict distograms with DMPfold2 from a sequence alignment.
To generate models, you will need to set up at least one of the following 3 approaches.
[See the DMPfold2 paper for a comparison of the approaches.]

### CNS

[CNS install instructions.]
[Source before run.]

### Xplor-NIH

[Xplor-NIH install instructions.]

### Force-directed folding

No extra setup is required.
However, it is strongly recommended that you have a GPU available as folding takes considerably longer on the CPU.

## Usage

Run `dmpfold -h` to see a help message.
To run DMPfold2 you will need the sequence alignment in `aln` format: one sequence per line with the ungapped target sequence as the first line.
Then run:
```bash
dmpfold -i prot.aln -o out_dir -p {fdf,cns,xplor}
```
where `out_dir` is the output directory to write results to (which will be created) and the folding protocol is selected from the list.

### Getting sequence alignments

Sequence alignments can be obtained from a target sequence in a number of ways, for example by running `hhblits` on the Uniclust database.
[Example script and conversion to aln.]

### Predicting distances

Residue-residue distances, hydrogen bonds and dihedral angles can be predicted using the Python interface:
```python
from dmpfold import aln_to_predictions, write_predictions
output = aln_to_predictions(aln_filepath)
write_predictions(output, "out")
```
You can also run structure prediction from within Python, for example:
```python
from dmpfold import aln_to_model_fdf
aln_to_model_fdf(aln_filepath, out_dir)
```
