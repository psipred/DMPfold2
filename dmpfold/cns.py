import sys
import os
import shutil
from datetime import datetime
from contextlib import redirect_stdout

from .networks import aln_to_predictions, aln_to_predictions_iter
from .utils import *

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

# Generate models with CNS
def generate_models_cns(output, bin_dir, target, iter_n, nmodels, contactperc, hbrange, hbprob):
    write_contact_constraints(output, "contacts.current", pthresh=contactperc)
    run(f"{bin_dir}/contact2noe {target}.fasta contacts.current > contact.tbl")

    write_hbond_constraints(output, "hbcontacts.current", topomin=hbrange, minprob=hbprob)
    run(f"{bin_dir}/hbond2noe hbcontacts.current > hbond.tbl")
    run(f"{bin_dir}/hbond2ssnoe hbcontacts.current > ssnoe.tbl")

    run(f"{cns_cmd} < dgsa.inp > dgsa.log")

    for model_i in range(nmodels):
        model_n = model_i + 1
        if not os.path.isfile(f"{target}_{model_n}.pdb"):
            raise Exception("CNS execution failed, check dgsa.log for details")
        with open(f"ensemble.{iter_n + 1}.pdb", "a") as of:
            for line in order_pdb_file(f"{target}_{model_n}.pdb"):
                of.write(line)
            of.write("END\n")
        os.remove(f"{target}_{model_n}.pdb")
        os.remove(f"{target}_sub_embed_{model_n}.pdb")

# Sample constraints and generate a single model with CNS
def sample_model_cns(output, bin_dir, target, iter_n):
    write_contact_constraints(output, "contacts.current", pthresh="random")
    run(f"{bin_dir}/contact2noe {target}.fasta contacts.current > contact.tbl")

    write_hbond_constraints(output, "hbcontacts.current", topomin="random", minprob="random")
    run(f"{bin_dir}/hbond2noe hbcontacts.current > hbond.tbl")
    run(f"{bin_dir}/hbond2ssnoe hbcontacts.current > ssnoe.tbl")

    run(f"{cns_cmd} < dgsa.inp > dgsa.log")

    if not os.path.isfile(f"{target}_1.pdb"):
        raise Exception("CNS execution failed, check dgsa.log for details")

    with open(f"ensemble.{iter_n + 1}.pdb", "a") as of:
        for line in order_pdb_file(f"{target}_1.pdb"):
            of.write(line)
        of.write("END\n")
    os.remove(f"{target}_1.pdb")
    os.remove(f"{target}_sub_embed_1.pdb")

# Protein structure prediction with CNS
def aln_to_model_cns(aln_filepath, out_dir, sample_relax=False,
                        relax_cmd="relax.static.linuxgccrelease", ncycles=-1, nmodels1=-1,
                        nmodels2=-1):
    if ncycles == -1:
        ncycles = 10 if sample_relax else 2
    if nmodels1 == -1:
        nmodels1 = 100
    if nmodels2 == -1:
        nmodels2 = 20

    phiprob1 = 0.88
    psiprob1 = 0.98
    phiprob2 = 0.88
    psiprob2 = 0.98
    if not sample_relax:
        contactperc1 = 0.42
        contactperc2 = 0.43
        hbprob1      = 0.85
        hbprob2      = 0.85
        hbrange      = 4

    start_time = datetime.now()
    print("Predicting structure from the alignment in", aln_filepath)

    with open(aln_filepath, "r") as f:
        aln = f.read().splitlines()
    sequence = aln[0]
    length = len(sequence)
    print("Sequence has", length, "residues:")
    print(sequence)
    print()
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

    print(f"Starting iteration 1 of {ncycles}")
    print()

    output = aln_to_predictions(os.path.join(cwd, aln_filepath))

    print("Neural network inference done, generating models")
    print()

    write_dihedral_constraints(output, "dihedral.tbl", "phi", phiprob1)
    write_dihedral_constraints(output, "dihedral.tbl", "psi", psiprob1)

    if sample_relax:
        write_dgsa_file(f"{cnsfile_dir}/dgsa.inp", "dgsa.inp", target, 1)
        for model_i in range(nmodels1):
            sample_model_cns(output, bin_dir, target, 0)
    else:
        write_dgsa_file(f"{cnsfile_dir}/dgsa.inp", "dgsa.inp", target, nmodels1)
        generate_models_cns(output, bin_dir, target, 0, nmodels1,
                            contactperc1, hbrange, hbprob1)

    run("./qmodope_mainens ensemble.1.pdb")
    print()

    for iter_n in range(1, ncycles):
        print(f"Starting iteration {iter_n + 1} of {ncycles}")
        print()

        shutil.move("contacts.current", f"contacts.{iter_n}")
        shutil.move("hbcontacts.current", f"hbcontacts.{iter_n}")

        if sample_relax:
            if iter_n >= 2:
                with open(os.devnull, "w") as f, redirect_stdout(f):
                    run(f"{relax_cmd} -overwrite -in:file:s best_qdope.pdb")
            else:
                shutil.copyfile("best_qdope.pdb", "best_qdope_0001.pdb")

            with open("best_qdope_0001.pdb") as f, open("ref.pdb", "w") as of:
                for line in f:
                    if line.startswith("ATOM"):
                        of.write(line)
                of.write("END\n")
            os.remove("best_qdope_0001.pdb")

            output = aln_to_predictions_iter(os.path.join(cwd, aln_filepath), "ref.pdb")
        else:
            output = aln_to_predictions_iter(os.path.join(cwd, aln_filepath), "best_qdope.pdb")

        print("Neural network inference done, generating models")
        print()

        write_dihedral_constraints(output, "dihedral.tbl", "phi", phiprob2)
        write_dihedral_constraints(output, "dihedral.tbl", "psi", psiprob2)

        if sample_relax:
            write_dgsa_file(f"{cnsfile_dir}/dgsa.inp", "dgsa.inp", target, 1)
            for model_i in range(nmodels2):
                sample_model_cns(output, bin_dir, target, iter_n)
        else:
            write_dgsa_file(f"{cnsfile_dir}/dgsa.inp", "dgsa.inp", target, nmodels2)
            generate_models_cns(output, bin_dir, target, iter_n, nmodels2,
                                contactperc2, hbrange, hbprob2)

        run(f"./qmodope_mainens ensemble.{iter_n + 1}.pdb")
        print()

    print("Clustering models")
    print()
    cluster_models(bin_dir, ncycles)
    print()

    for modcheck_file in modcheck_files:
        os.remove(modcheck_file)

    os.chdir(cwd)
    print("Writing output to", out_dir)
    print("Done in", str(datetime.now() - start_time).split(".")[0])
