import sys
import os
import shutil
import subprocess
from random import random, randrange
from math import pi, degrees, atan2
from operator import itemgetter

import torch
import numpy as np

from .networks import n_bins

one_to_three_aas = {"C": "CYS", "D": "ASP", "S": "SER", "Q": "GLN", "K": "LYS",
                    "I": "ILE", "P": "PRO", "T": "THR", "F": "PHE", "N": "ASN",
                    "G": "GLY", "H": "HIS", "L": "LEU", "R": "ARG", "W": "TRP",
                    "A": "ALA", "V": "VAL", "E": "GLU", "Y": "TYR", "M": "MET"}

modcheck_files = ("dope.scr", "qmodcheck", "qmodope_mainens", "modcheckpot.dat")

# Run a command and exit with an error if the return code is not 0
def run(cmd):
    proc = subprocess.run(cmd, shell=True)
    if proc.returncode != 0:
        print(f"Command `{cmd}` gave return code {proc.returncode}, exiting")
        sys.exit(1)

# Random number as called with $RANDOM in Bash
def random_seed():
    return randrange(0, 32768)

# Write sequence file for CNS
def write_seq_file(in_file, out_file):
    seq = ""
    with open(in_file) as f:
        for line in f:
            if not line.startswith(">"):
                seq += line.rstrip()
    with open(out_file, "w") as of:
        for ri, res in enumerate(seq):
            of.write(one_to_three_aas[res])
            if ri % 12 == 11:
                of.write("\n")
            else:
                of.write(" ")

# Append dihedral CNS constraints to a file
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

# Write hydrogen bond CNS constraints to a file
def write_hbond_constraints(output, out_file, topomin, minprob):
    if topomin == "random":
        topomin = randrange(0, 6) + 2
    if minprob == "random":
        minprob = random() * 0.5 + 0.3
    length = output.size(2)
    with open(out_file, "w") as of:
        for wi in range(length):
            for wj in range(length):
                if abs(wi - wj) >= topomin:
                    probs = output.data[0, 0:2, wi, wj]
                    score = torch.sum(probs[1:2])
                    if score > minprob:
                        of.write(f"{wi + 1} {wj + 1} 0 3.5 {score}\n")

# Write contact CNS constraints to a file
def write_contact_constraints(output, out_file, pthresh):
    if pthresh == "random":
        pthresh = random() * 0.6 + 0.3
    length = output.size(2)
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

# Order atoms within each residue in a PDB file
def order_pdb_file(pdb_file):
    lines = []
    res_lines = []
    atom_number = 1
    with open(pdb_file) as f:
        for line in f:
            if line.startswith("ATOM") and line[12] != "H" and line[13] != "H":
                resnum = int(line[22:26])
                if len(res_lines) == 0 and atom_number == 1:
                    last_resnum = resnum
                if resnum != last_resnum:
                    for res_line, sorting_char in sorted(res_lines, key=itemgetter(1)):
                        lines.append(res_line[:6] + str(atom_number).rjust(5) + res_line[11:])
                        atom_number += 1
                    res_lines = []
                atom_name = line[12:16]
                if atom_name == " N  ":
                    sorting_char = "1"
                elif atom_name == " CA ":
                    sorting_char = "2"
                elif atom_name == " C  ":
                    sorting_char = "3"
                elif atom_name == " O  ":
                    sorting_char = "4"
                elif atom_name == " OXT":
                    sorting_char = "ZZ"
                else:
                    sorting_char = atom_name[2:].translate(str.maketrans("GDEZH", "CDEFG"))
                res_lines.append((line, sorting_char))
        for res_line, sorting_char in sorted(res_lines, key=itemgetter(1)):
            lines.append(res_line[:6] + str(atom_number).rjust(5) + res_line[11:])
            atom_number += 1
    return lines

# Combine ensembles from iterations, cluster, and write 1-5 output files
def cluster_models(bin_dir, ncycles):
    with open("ensemble.pdb", "w") as of:
        for iter_n in range(ncycles):
            with open(f"ensemble.{iter_n + 1}.pdb") as f:
                of.write(f.read())
    run(f"{bin_dir}/tmclust ensemble.pdb")

    if os.path.isfile("CLUSTER_001.pdb"):
        run("./qmodope_mainens CLUSTER_001.pdb")
    else:
        run("./qmodope_mainens ensemble.pdb")
    shutil.move("best_qdope.pdb", "final_1.pdb")

    for cn in range(2, 6):
        if os.path.isfile(f"CLUSTER_00{cn}.pdb"):
            run(f"./qmodope_mainens CLUSTER_00{cn}.pdb")
            shutil.move("best_qdope.pdb", f"final_{cn}.pdb")
