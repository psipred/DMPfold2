import os
from random import random, randrange
from math import pi, degrees, atan2
from subprocess import run

import numpy as np

from .networks import aln_to_predictions, aln_to_predictions_iter

ncycles = 2 # Number of cycles
nmodels1 = 100 # Number of models in first cycle
nmodels2 = 20 # Number of models in subsequence cycles

phiprob1 = 0.88
psiprob1 = 0.98
phiprob2 = 0.88
psiprob2 = 0.98

n_bins = 34

# Write CNS folding script
def write_dgsa_file(in_file, out_file, target, n_models):
    seed = randrange(0, 32768)
    with open(in_file) as f, open(out_file, "w") as of:
        text = f.read()
        text = text.replace("_TARGET_NAME_", target       )
        text = text.replace("_SEED_"       , str(seed)    )
        text = text.replace("_NMODELS_"    , str(n_models))
        of.write(text)

# Append dihedral constraints to a file
def write_dihedral_constraints(output, out_file, angle, prob):
    n_res = output.size(2)
    for wi in range(n_res - 1):
        if angle == "phi":
            preds = output.data[0, 36:70 , wi, wi + 1].clone()
        elif angle == "psi":
            preds = output.data[0, 70:104, wi, wi + 1].clone()
        ang, sdev, meets_threshold = dihedral_bins_to_constraints(preds, prob)
        if meets_threshold:
            with open(out_file, "a") as of:
                i, j = wi + 1, wi + 2
                if angle == "phi":
                    of.write((f"assign (resid {i} and name c) (resid {j} and name n) (resid {j} and name ca) "
                                f"(resid {j} and name c) 1.0 {ang} {sdev} 2\n"))
                elif angle == "psi":
                    of.write((f"assign (resid {i} and name n) (resid {i} and name ca) (resid {i} and name c) "
                                f"(resid {j} and name n) 1.0 {ang} {sdev} 2\n"))

# Convert predicted dihedrals from bins to constraints
def dihedral_bins_to_constraints(fields, pthresh):
    pmax = 0.0
    for k in range(n_bins):
        if fields[k] > pmax:
            pmax = fields[k]
            kmax = k
    psum = pmax
    k1 = kmax - 1
    k2 = kmax + 1
    lastk1, lastk2 = kmax, kmax
    while psum < pthresh and (k2 - k1) < (n_bins - 1):
        p1 = fields[k1 % n_bins]
        p2 = fields[k2 % n_bins]
        if p1 > p2:
            psum += p1
            lastk1 = k1
            k1 -= 1
        else:
            psum += p2
            lastk2 = k2
            k2 += 1
    sinsum, cossum = 0.0, 0.0
    for k in range(n_bins):
        sinsum += np.sin((k + 0.5) * 2 * pi / n_bins - pi) * fields[k]
        cossum += np.cos((k + 0.5) * 2 * pi / n_bins - pi) * fields[k]
    ang = degrees(atan2(sinsum, cossum))
    sdev = (abs(lastk2 - lastk1) + 1) * 360.0 / (n_bins * 2)
    meets_threshold = psum > pthresh
    return ang, sdev, meets_threshold

def write_hbond_constraints(output, out_file):
    pass

def write_contact_constraints(output, out_file):
    length = output.size(2)
    pthresh = random() * 0.6 + 0.3
    with open(out_file, "w") as of:
        for wi in range(0, length - 5):
            for wj in range(wi + 5, length):
                fields = output.data[0, 2:36, wi, wj]
                pmax = 0.0
                for k in range(n_bins):
                    if fields[k] > pmax:
                        pmax = fields[k]
                        kmax = k
                if kmax < (n_bins - 1):
                    psum = pmax
                    k1 = kmax - 1
                    k2 = kmax + 1
                    lastk1, lastk2 = kmax, kmax
                    while 0.0 < psum < pthresh and (k1 >= 0 or k2 < (n_bins - 1)):
                        if k1 >= 0:
                            p1 = fields[k1]
                        else:
                            p1 = -1.0
                        if k2 < (n_bins - 1):
                            p2 = fields[k2]
                        else:
                            p2 = -1.0

                        if p1 >= p2:
                            psum += p1
                            lastk1 = k1
                            k1 -= 1
                        else:
                            psum += p2
                            lastk2 = k2
                            k2 += 1
                    if psum >= pthresh:
                        dmin = 3.5 + 0.5 * lastk1
                        dmax = 4.0 + 0.5 * lastk2
                        of.write(f"{wi + 1} {wj + 1} {dmin} {dmax} {psum}\n")

def aln_to_model_cns(aln_filepath, out_dir):
    with open(aln_filepath, "r") as f:
        aln = f.read().splitlines()
    sequence = aln[0]
    length = len(sequence)
    print("Sequence has", length, "residues:")
    print(sequence)
    target = os.path.split(aln_filepath)[1].rsplit(".", 1)[0]

    with open("input.fasta", "w") as f:
        f.write(">SEQ\n")
        f.write(sequence + "\n")
    bin_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "bin")
    run(f"{bin_dir}/fasta2tlc < input.fasta > input.seq", shell=True)

    cns_cmd = "cns"
    cnsfile_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "cnsfiles")
    run(f"{cns_cmd} < {cnsfile_dir}/gseq.inp > gseq.log", shell=True)
    run(f"{cns_cmd} < {cnsfile_dir}/extn.inp > extn.log", shell=True)

    output = aln_to_predictions(aln_filepath)

    write_dihedral_constraints(output, "dihedral.tbl", "phi", phiprob1)
    write_dihedral_constraints(output, "dihedral.tbl", "psi", psiprob1)

    write_dgsa_file(f"{cnsfile_dir}/dgsa.inp", "dgsa.inp", target, 1)

    write_hbond_constraints(output, "hbcontacts.current")
    run(f"{bin_dir}/hbond2noe hbcontacts.current > hbond.tbl", shell=True)
    run(f"{bin_dir}/hbond2ssnoe hbcontacts.current > ssnoe.tbl", shell=True)

    write_contact_constraints(output, "contact.tbl")

    run("cns < dgsa.inp > dgsa.log", shell=True)

    run("./qmodope_mainens ensemble.1.pdb", shell=True)

    for iter_n in range(1, ncycles):
        output = aln_to_predictions_iter(aln_filepath, "best_qdope.pdb")

        write_dihedral_constraints(output, "dihedral.tbl", "phi", phiprob2)
        write_dihedral_constraints(output, "dihedral.tbl", "psi", psiprob2)

        write_dgsa_file(f"{cnsfile_dir}/dgsa.inp", "dgsa.inp", target, 1)

        write_hbond_constraints(output, "hbcontacts.current")
        run(f"{bin_dir}/hbond2noe hbcontacts.current > hbond.tbl", shell=True)
        run(f"{bin_dir}/hbond2ssnoe hbcontacts.current > ssnoe.tbl", shell=True)

        write_contact_constraints(output, "contact.tbl")

        run("cns < dgsa.inp > dgsa.log", shell=True)

        run(f"./qmodope_mainens ensemble.{iter_n}.pdb", shell=True)

        run(f"{bin_dir}/tmclust ensemble.pdb", shell=True)
