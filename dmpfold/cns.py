import os
from random import randrange
from subprocess import run

ncycles = 2 # Number of cycles
nmodels1 = 100 # Number of models in first cycle
nmodels2 = 20 # Number of models in subsequence cycles

phiprob1 = 0.88
psiprob1 = 0.98
phiprob2 = 0.88
psiprob2 = 0.98

def write_dgsa_file(in_file, out_file, target, n_models):
    seed = randrange(0, 32768)
    with open(in_file) as f, open(out_file, "w") as of:
        text = r.read()
        text = text.replace("_TARGET_NAME_", target       )
        text = text.replace("_SEED_"       , str(seed)    )
        text = text.replace("_NMODELS_"    , str(n_models))
        of.write(text)

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

    write_dihedral_contsraints(output, "dihedral.tbl", "phi", phiprob1)
    write_dihedral_contsraints(output, "dihedral.tbl", "psi", psiprob1)

    write_dgsa_file(f"{cnsfile_dir}/dgsa.inp", "dgsa.inp", target, 1)

    write_hbond_constraints("hbond.tbl", "ssnoe.tbl")
    write_contact_constraints("contact.tbl")

    run("cns < dgsa.inp > dgsa.log", shell=True)

    run("./qmodope_mainens ensemble.1.pdb", shell=True)

    for iter_n in range(1, ncycles):
        output = aln_to_predictions_iter(aln_filepath, "best_qdope.pdb")

        write_dihedral_contsraints(output, "dihedral.tbl", "phi", phiprob2)
        write_dihedral_contsraints(output, "dihedral.tbl", "psi", psiprob2)

        write_dgsa_file(f"{cnsfile_dir}/dgsa.inp", "dgsa.inp", target, 1)

        write_hbond_constraints("hbond.tbl", "ssnoe.tbl")
        write_contact_constraints("contact.tbl")

        run("cns < dgsa.inp > dgsa.log", shell=True)

        run(f"./qmodope_mainens ensemble.{iter_n}.pdb", shell=True)

        run(f"{bin_dir}/tmclust ensemble.pdb", shell=True)
