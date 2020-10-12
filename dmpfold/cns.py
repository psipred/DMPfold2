import sys
import os
import shutil
from datetime import datetime

from .networks import aln_to_predictions, aln_to_predictions_iter
from .utils import *

ncycles = 2 # Number of cycles
nmodels1 = 100 # Number of models in first cycle
nmodels2 = 20 # Number of models in subsequence cycles

phiprob1 = 0.88
psiprob1 = 0.98
phiprob2 = 0.88
psiprob2 = 0.98

cns_cmd = "cns"

# Write CNS folding script
def write_dgsa_file(in_file, out_file, target, n_models):
    seed = random_seed()
    with open(in_file) as f, open(out_file, "w") as of:
        text = f.read()
        text = text.replace("_TARGET_NAME_", target       )
        text = text.replace("_SEED_"       , str(seed)    )
        text = text.replace("_NMODELS_"    , str(n_models))
        of.write(text)

# Sample constraints and generate a single model with CNS
def generate_model_cns(output, bin_dir, target, iter_n):
    write_hbond_constraints(output, "hbcontacts.current", topomin="random", minprob="random")
    run(f"{bin_dir}/hbond2noe hbcontacts.current > hbond.tbl")
    run(f"{bin_dir}/hbond2ssnoe hbcontacts.current > ssnoe.tbl")

    write_contact_constraints(output, "contacts.current", pthresh="random")
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
        sys.exit(1)
    else:
        cwd = os.getcwd()
        os.mkdir(out_dir)
        os.chdir(out_dir)

    dmpfold_dir  = os.path.dirname(os.path.realpath(__file__))
    bin_dir      = os.path.join(dmpfold_dir, "bin")
    cnsfile_dir  = os.path.join(dmpfold_dir, "cnsfiles")
    modcheck_dir = os.path.join(dmpfold_dir, "modcheck")

    for modcheck_file in modcheck_files:
        os.symlink(f"{modcheck_dir}/{modcheck_file}", modcheck_file)

    with open(f"{target}.fasta", "w") as f:
        f.write(">SEQ\n")
        f.write(sequence + "\n")
    write_seq_file(f"{target}.fasta", "input.seq")

    run(f"{cns_cmd} < {cnsfile_dir}/gseq.inp > gseq.log")
    run(f"{cns_cmd} < {cnsfile_dir}/extn.inp > extn.log")

    output = aln_to_predictions(os.path.join(cwd, aln_filepath))

    write_dihedral_constraints(output, "dihedral.tbl", "phi", phiprob1)
    write_dihedral_constraints(output, "dihedral.tbl", "psi", psiprob1)

    write_dgsa_file(f"{cnsfile_dir}/dgsa.inp", "dgsa.inp", target, 1)

    for model_n in range(nmodels1):
        generate_model_cns(output, bin_dir, target, 0)

    run("./qmodope_mainens ensemble.1.pdb")

    for iter_n in range(1, ncycles):
        shutil.move("contacts.current", f"contacts.{iter_n}")
        shutil.move("hbcontacts.current", f"hbcontacts.{iter_n}")

        output = aln_to_predictions_iter(os.path.join(cwd, aln_filepath), "best_qdope.pdb")

        write_dihedral_constraints(output, "dihedral.tbl", "phi", phiprob2)
        write_dihedral_constraints(output, "dihedral.tbl", "psi", psiprob2)

        write_dgsa_file(f"{cnsfile_dir}/dgsa.inp", "dgsa.inp", target, 1)

        for model_n in range(nmodels2):
            generate_model_cns(output, bin_dir, target, iter_n)

        run(f"./qmodope_mainens ensemble.{iter_n + 1}.pdb")

    cluster_models(bin_dir, ncycles)

    for modcheck_file in modcheck_files:
        os.remove(modcheck_file)

    os.chdir(cwd)
    print("Writing output to", out_dir)
    print("Done in", str(datetime.now() - start_time).split(".")[0])
