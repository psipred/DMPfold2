import sys
import os
import shutil
from datetime import datetime
from glob import glob

from .networks import aln_to_predictions, aln_to_predictions_iter
from .utils import *

contactperc1 = 0.47
contactperc2 = 0.59
hbprob1 = 0.49
hbprob2 = 0.72
phiprob1 = 0.25
psiprob1 = 0.91
phiprob2 = 0.16
psiprob2 = 0.68
hbrange = 3

# Scale factor and exponent for the DG restraint energy term for regularisation
xn_mmdgscale = 100.0
xn_mmdgexp = 2

# Generate models with Xplor-NIH
def generate_models_xplor(output, xplor_bin_dir, xplor_script_dir, ncpus, iter_n, nmodels):
    nnoe = 1
    for fp in ("contact.tbl", "hbond.tbl", "ssnoe.tbl"):
        with open(fp) as f:
            nnoe += len(f.readlines())

    with open("dihedral.tbl") as f:
        ndih = len(f.readlines()) + 1

    for fp in glob("dg*.pdb*"):
        os.remove(fp)

    seed = random_seed()
    with open(f"{xplor_script_dir}/nmr___dg_sub_embed.inp") as f, open("dgsub.inp", "w") as of:
        text = f.read()
        text = text.replace("__XPLORSEED__", str(seed)        )
        text = text.replace("__NMODELS__"  , str(nmodels)     )
        text = text.replace("__NNOE__"     , str(nnoe)        )
        text = text.replace("__NDIH__"     , str(ndih)        )
        text = text.replace("_MMDGEXP_"    , str(xn_mmdgexp)  )
        text = text.replace("_MMDGSCALE_"  , str(xn_mmdgscale))
        of.write(text)

    xn_cpu = min(nmodels, ncpus)
    run(f"{xplor_bin_dir}/xplor -smp {xn_cpu} -omp 1 -o dg_sub.log dgsub.inp")

    with open(f"{xplor_script_dir}/nmr___dgsa.inp") as f, open("dgsa.inp", "w") as of:
        text = f.read()
        text = text.replace("__XPLORSEED__", str(seed)        )
        text = text.replace("__NMODELS__"  , str(nmodels)     )
        text = text.replace("__NNOE__"     , str(nnoe)        )
        text = text.replace("__NDIH__"     , str(ndih)        )
        of.write(text)

    run(f"{xplor_bin_dir}/xplor -smp {xn_cpu} -omp 1 -o dgsa.log dgsa.inp")

    with open(f"ensemble.{iter_n + 1}.pdb", "a") as of:
        for fp in glob("dgsa_[0-9]*.pdb"):
            for line in order_pdb_file(fp):
                of.write(line)
            of.write("END\n")

# Protein structure prediction with Xplor-NIH
def aln_to_model_xplor(aln_filepath, out_dir, xplor_bin_dir, ncpus=4,
                        ncycles=-1, nmodels1=-1, nmodels2=-1):
    if ncycles == -1:
        ncycles = 2
    if nmodels1 == -1:
        nmodels1 = 20
    if nmodels2 == -1:
        nmodels2 = 20

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

    dmpfold_dir      = os.path.dirname(os.path.realpath(__file__))
    bin_dir          = os.path.join(dmpfold_dir, "bin")
    xplor_script_dir = os.path.join(dmpfold_dir, "xplorfiles")
    modcheck_dir     = os.path.join(dmpfold_dir, "modcheck")

    for modcheck_file in modcheck_files:
        os.symlink(f"{modcheck_dir}/{modcheck_file}", modcheck_file)

    with open(f"{target}.fasta", "w") as f:
        f.write(">SEQ\n")
        f.write(sequence + "\n")
    write_seq_file(f"{target}.fasta", "input.seq")

    run(f"{xplor_bin_dir}/seq2psf input.seq > seq2psf.log")

    seed = random_seed()
    with open(f"{xplor_script_dir}/nmr___generate_template.inp") as f, open("gt.inp", "w") as of:
        text = f.read()
        text = text.replace("__XPLORSEED__", str(seed))
        of.write(text)

    run(f"{xplor_bin_dir}/xplor < gt.inp > gen_templ.log")

    print(f"Starting iteration 1 of {ncycles}")
    print()

    output = aln_to_predictions(os.path.join(cwd, aln_filepath))

    print("Neural network inference done, generating models")
    print()

    write_contact_constraints(output, "contacts.current", pthresh=contactperc1)

    write_hbond_constraints(output, "hbcontacts.current", topomin=hbrange, minprob=hbprob1)

    write_dihedral_constraints(output, "dihedral.tbl", "phi", phiprob1)
    write_dihedral_constraints(output, "dihedral.tbl", "psi", psiprob1)

    run(f"{bin_dir}/contact2noe {target}.fasta contacts.current > contact.tbl")
    run(f"{bin_dir}/hbond2noe hbcontacts.current > hbond.tbl")
    run(f"{bin_dir}/hbond2ssnoe hbcontacts.current > ssnoe.tbl")

    generate_models_xplor(output, xplor_bin_dir, xplor_script_dir, ncpus, 0, nmodels1)
    print()

    run("./qmodope_mainens ensemble.1.pdb")
    print()

    for iter_n in range(1, ncycles):
        print(f"Starting iteration {iter_n + 1} of {ncycles}")
        print()

        shutil.move("contacts.current", f"contacts.{iter_n}")
        shutil.move("hbcontacts.current", f"hbcontacts.{iter_n}")

        output = aln_to_predictions_iter(os.path.join(cwd, aln_filepath), "best_qdope.pdb")

        print("Neural network inference done, generating models")
        print()

        write_contact_constraints(output, "contacts.current", pthresh=contactperc2)

        write_hbond_constraints(output, "hbcontacts.current", topomin=hbrange, minprob=hbprob2)

        write_dihedral_constraints(output, "dihedral.tbl", "phi", phiprob2)
        write_dihedral_constraints(output, "dihedral.tbl", "psi", psiprob2)

        run(f"{bin_dir}/contact2noe {target}.fasta contacts.current > contact.tbl")
        run(f"{bin_dir}/hbond2noe hbcontacts.current > hbond.tbl")
        run(f"{bin_dir}/hbond2ssnoe hbcontacts.current > ssnoe.tbl")

        generate_models_xplor(output, xplor_bin_dir, xplor_script_dir, ncpus, iter_n, nmodels2)
        print()

        run(f"./qmodope_mainens ensemble.{iter_n + 1}.pdb")
        print()

    for fp in glob(".xplorInput*"):
        os.remove(fp)

    print("Clustering models")
    print()
    cluster_models(bin_dir, ncycles)
    print()

    for modcheck_file in modcheck_files:
        os.remove(modcheck_file)

    os.chdir(cwd)
    print("Writing output to", out_dir)
    print("Done in", str(datetime.now() - start_time).split(".")[0])
