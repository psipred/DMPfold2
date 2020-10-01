# DMPfold2

Development code for DMPfold2 distribution (currently private).

## Installation

We have attempted to make DMPfold2 easier to install than DMPfold1, which had many more dependencies.

1. Python3 is required.
2. Install [PyTorch](https://pytorch.org) as appropriate for your system. It is not necessary to have a GPU, though it is recommended if using the force-directed folding protocol (see below).
3. Install [MODELLER](https://salilab.org/modeller), which is usually as simple as running `conda install -c salilab modeller` and modifying the license key.
4. To install DMPfold2, run:
```
git clone https://github.com/psipred/DMPfold2
cd DMPfold2
pip install -e .
```
[This will become `pip install dmpfold` on registering in PyPI.]
This will automatically install NumPy, SciPy and PeptideBuilder if required.

The above steps are sufficient to predict distograms with DMPfold2.
To generate models, you will need to set up at least one of the following 3 approaches.
[See the DMPfold2 paper for a comparison of the approaches.]

### CNS

cns

### Xplor-NIH

xplor

### Force-directed folding

gpu
