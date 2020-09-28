import os
from subprocess import run

def aln_to_model_cns(aln_filepath, out_dir):
    with open(aln_filepath, "r") as f:
        aln = f.read().splitlines()
    sequence = aln[0]
    length = len(sequence)
    print("Sequence has", length, "residues:")
    print(sequence)

    with open("input.fasta", "w") as f:
        f.write(">SEQ\n")
        f.write(sequence + "\n")
    bin_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "bin")
    run(f"{bin_dir}/fasta2tlc < input.fasta > input.seq", shell=True)

    cns_cmd = "cns"
    cnsfile_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "cnsfiles")
    run(f"{cns_cmd} < {cnsfile_dir}/gseq.inp > gseq.log", shell=True)
    run(f"{cns_cmd} < {cnsfile_dir}/extn.inp > extn.log", shell=True)
