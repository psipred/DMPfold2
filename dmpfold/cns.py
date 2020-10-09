import sys
import os
import shutil
import subprocess
from random import random, randrange
from math import pi, degrees, atan2
from datetime import datetime
from operator import itemgetter

import torch
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
modcheck_files = ("dope.scr", "qmodcheck", "qmodope_mainens", "modcheckpot.dat")
cns_cmd = "cns"

def run(cmd):
    proc = subprocess.run(cmd, shell=True)
    if proc.returncode != 0:
        print(f"Command `{cmd}` gave return code {proc.returncode}, exiting")
        sys.exit()

# Write CNS folding script
def write_dgsa_file(in_file, out_file, target, n_models):
    seed = randrange(0, 32768)
    with open(in_file) as f, open(out_file, "w") as of:
        text = f.read()
        text = text.replace("_TARGET_NAME_", target       )
        text = text.replace("_SEED_"       , str(seed)    )
        text = text.replace("_NMODELS_"    , str(n_models))
        of.write(text)

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
def write_hbond_constraints(output, out_file):
    length = output.size(2)
    topomin = randrange(0, 6) + 2
    minprob = random() * 0.5 + 0.3
    with open(out_file, "w") as of:
        for wi in range(length):
            for wj in range(length):
                if abs(wi - wj) >= topomin:
                    probs = output.data[0, 0:2, wi, wj]
                    score = torch.sum(probs[1:2])
                    if score > minprob:
                        of.write(f"{wi + 1} {wj + 1} 0 3.5 {score}\n")

# Write contact CNS constraints to a file
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

# Sample constraints and generate a single model with CNS
def generate_model(output, bin_dir, target, iter_n):
    write_hbond_constraints(output, "hbcontacts.current")
    run(f"{bin_dir}/hbond2noe hbcontacts.current > hbond.tbl")
    run(f"{bin_dir}/hbond2ssnoe hbcontacts.current > ssnoe.tbl")

    write_contact_constraints(output, "contacts.current")
    run(f"{bin_dir}/contact2noe {target}.fasta contacts.current > contact.tbl")

    run(f"{cns_cmd} < dgsa.inp > dgsa.log")

    with open(f"ensemble.{iter_n + 1}.pdb", "a") as of:
        for line in order_pdb_file(f"{target}_1.pdb"):
            of.write(line)
        of.write("END\n")
    os.remove(f"{target}_1.pdb")
    os.remove(f"{target}_sub_embed_1.pdb")

# Protein structure prediction with CNS
def aln_to_model_cns(aln_filepath, out_dir):
    start_time = datetime.now()
    print("Predicting structure from the alignment in", aln_filepath)

    with open(aln_filepath, "r") as f:
        aln = f.read().splitlines()
    sequence = aln[0]
    length = len(sequence)
    print("Sequence has", length, "residues:")
    print(sequence)
    target = os.path.split(aln_filepath)[1].rsplit(".", 1)[0]

    if os.path.isdir(out_dir):
        print(f"Output directory {out_dir} already exists, exiting")
        sys.exit()
    else:
        cwd = os.getcwd()
        os.mkdir(out_dir)
        os.chdir(out_dir)

    dmpfold_dir  = os.path.dirname(os.path.realpath(__file__))
    bin_dir      = os.path.join(dmpfold_dir, "bin")
    cnsfile_dir  = os.path.join(dmpfold_dir, "cnsfiles")
    modcheck_dir = os.path.join(dmpfold_dir, "modcheck")

    with open(f"{target}.fasta", "w") as f:
        f.write(">SEQ\n")
        f.write(sequence + "\n")
    run(f"{bin_dir}/fasta2tlc < {target}.fasta > input.seq")

    run(f"{cns_cmd} < {cnsfile_dir}/gseq.inp > gseq.log")
    run(f"{cns_cmd} < {cnsfile_dir}/extn.inp > extn.log")

    for modcheck_file in modcheck_files:
        os.symlink(f"{modcheck_dir}/{modcheck_file}", modcheck_file)

    output = aln_to_predictions(os.path.join(cwd, aln_filepath))

    write_dihedral_constraints(output, "dihedral.tbl", "phi", phiprob1)
    write_dihedral_constraints(output, "dihedral.tbl", "psi", psiprob1)

    write_dgsa_file(f"{cnsfile_dir}/dgsa.inp", "dgsa.inp", target, 1)

    for model_n in range(nmodels1):
        generate_model(output, bin_dir, target, 0)

    run("./qmodope_mainens ensemble.1.pdb")

    for iter_n in range(1, ncycles):
        shutil.move("contacts.current", f"contacts.{iter_n}")
        shutil.move("hbcontacts.current", f"hbcontacts.{iter_n}")

        output = aln_to_predictions_iter(os.path.join(cwd, aln_filepath), "best_qdope.pdb")

        write_dihedral_constraints(output, "dihedral.tbl", "phi", phiprob2)
        write_dihedral_constraints(output, "dihedral.tbl", "psi", psiprob2)

        write_dgsa_file(f"{cnsfile_dir}/dgsa.inp", "dgsa.inp", target, 1)

        for model_n in range(nmodels2):
            generate_model(output, bin_dir, target, iter_n)

        run(f"./qmodope_mainens ensemble.{iter_n + 1}.pdb")

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

    for modcheck_file in modcheck_files:
        os.remove(modcheck_file)

    os.chdir(cwd)
    print("Writing output to", out_dir)
    print("Done in", str(datetime.now() - start_time).split(".")[0])
