# ############################## DMPfold2 Main program ################################
# By David T. Jones, June 2021


from __future__ import print_function

import sys
import os
import time
import random
import argparse

from math import sqrt, log, asin, cos, pi, sin

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import normalize

from .network import GRUResNet


# reweight MSA based on cutoff (from https://github.com/gjoni/trRosetta/blob/master/network/utils.py)
def reweight(msa1hot, cutoff):
    id_min = msa1hot.shape[1] * cutoff
    id_mtx = torch.einsum('ikl,jkl->ij', msa1hot, msa1hot)
    id_mask = id_mtx > id_min
    w = 1. / id_mask.float().sum(dim=-1)
    return w


# shrunk covariance inversion (from https://github.com/gjoni/trRosetta/blob/master/network/utils.py)
def fast_dca(msa1hot, weights, penalty = 4.5):
    device = msa1hot.device
    nr, nc, ns = msa1hot.shape
    x = msa1hot.view(nr, -1)
    num_points = weights.sum() - torch.sqrt(weights.mean())

    mean = (x * weights[:, None]).sum(dim=0, keepdims=True) / num_points
    x = (x - mean) * torch.sqrt(weights[:, None])

    cov = (x.t() @ x) / num_points
    cov_reg = cov + torch.eye(nc * ns, device=device) * penalty / torch.sqrt(weights.sum())

    inv_cov = torch.inverse(cov_reg)
    x1 = inv_cov.view(nc, ns, nc, ns)
    x2 = x1.transpose(1, 2).contiguous()
    features = x2.reshape(nc, nc, ns * ns)

    x3 = torch.sqrt((x1[:, :-1, :, :-1] ** 2).sum(dim=(1, 3))) * (1 - torch.eye(nc, device=device))
    apc = x3.sum(dim=0, keepdims=True) * x3.sum(dim=1, keepdims=True) / x3.sum()
    contacts = (x3 - apc) * (1 - torch.eye(nc, device=device))
    return torch.cat((features, contacts[:, :, None]), dim=2)


def run_dmpfold():

    # Create the parser
    parser = argparse.ArgumentParser()
    # Add arguments
    parser.add_argument('-i', '--input_file', type=str, required=True)
    parser.add_argument('-d', '--device', type=str, default='cpu', required=False)
    parser.add_argument('-t', '--template', type=str, required=False)
    parser.add_argument('-n', '--iterations', type=int, default=10, required=False)
    parser.add_argument('-m', '--minsteps', type=int, default=100, required=False)
    # Parse the argument
    args = parser.parse_args()

    device = torch.device(args.device)

    # Create neural network model (depending on first command line parameter)
    network = GRUResNet(512,128).eval().to(device)

    modeldir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'trained_model')

    # Model parameters stored as two files to get round GitHub's file size limit
    trained_model = torch.load(os.path.join(modeldir, 'FINAL_fullmap_e2e_model_part1.pt'),
                                map_location=lambda storage, loc: storage)
    trained_model.update(torch.load(os.path.join(modeldir, 'FINAL_fullmap_e2e_model_part2.pt'),
                                map_location=lambda storage, loc: storage))
    network.load_state_dict(trained_model)

    aln = []
    with open(args.input_file, 'r') as alnfile:
        for line in alnfile.readlines():
            if not line.startswith(">"):
                aln.append(line.rstrip())

    if args.template is not None:
        with open(args.template, 'r') as tpltpdbfile:
            coords = []
            n = 0
            for line in tpltpdbfile:
                if line[:4] == 'ATOM' and line[12:16] == ' CA ':
                    # Split the line
                    pdb_fields = [line[:6], line[6:11], line[12:16], line[17:20], line[21],
                                  line[22:26], line[30:38], line[38:46], line[46:54]]
                    coords.append(np.array([float(pdb_fields[6]), float(pdb_fields[7]), float(pdb_fields[8])], dtype=np.float32))

        init_coords = torch.from_numpy(np.asarray(coords)).unsqueeze(0).to(device)
    else:
        init_coords = None

    nloops = args.iterations
    refine_steps = args.minsteps

    aa_trans = str.maketrans('ARNDCQEGHILKMFPSTWYVBJOUXZ-.', 'ABCDEFGHIJKLMNOPQRSTUUUUUUVV')

    nseqs = len(aln)
    length = len(aln[0])
    alnmat = (np.frombuffer(''.join(aln).translate(aa_trans).encode('latin-1'), dtype=np.uint8) - ord('A')).reshape(nseqs,length)

    if nseqs > 3000:
        alnmat = alnmat[:3000]
        nseqs = 3000

    inputs = torch.from_numpy(alnmat).type(torch.LongTensor).to(device)

    msa1hot = F.one_hot(torch.clamp(inputs, max=20), 21).float()
    w = reweight(msa1hot, cutoff=0.8)

    f2d_dca = fast_dca(msa1hot, w).float() if nseqs > 1 else torch.zeros((length, length, 442), device=device) 
    f2d_dca = f2d_dca.permute(2,0,1).unsqueeze(0)

    if init_coords is not None:
        dmap = (init_coords - init_coords.transpose(0,1)).pow(2).sum(dim=2).sqrt().unsqueeze(0).unsqueeze(0)
    else:
        dmap = torch.zeros((1, 1, length, length), device=device) - 1

    inputs2 = torch.cat((f2d_dca, dmap), dim=1)

    rnamedict = {
        0:'ALA', 1:'ARG', 2:'ASN', 3:'ASP', 4:'CYS', 5:'GLN', 6:'GLU', 7:'GLY', 8:'HIS',
        9:'ILE', 10:'LEU', 11:'LYS', 12:'MET', 13:'PHE', 14:'PRO', 15:'SER', 16:'THR', 17:'TRP',
        18:'TYR', 19:'VAL'
    }

    network.eval()
    with torch.no_grad():
        coords, confs = network(inputs, inputs2, nloops, refine_steps)
        coords = coords.view(-1,length,5,3)
        confs = confs[0]

    print("REMARK  CONF: ", confs.mean().item())
    atoms = (" N  ", " CA ", " C  ", " O  ", " CB ")
    atomnum = 1
    for ri in range(length):
        for ai, an in enumerate(atoms):
            if alnmat[0,ri] != 7 or ai != 4:
                print("ATOM   %4d %s %s  %4d    %8.3f%8.3f%8.3f  1.00%6.2f" % (
                    atomnum, an, rnamedict[alnmat[0,ri]], ri + 1,
                    coords[0, ri, ai, 0].item(),
                    coords[0, ri, ai, 1].item(),
                    coords[0, ri, ai, 2].item(),
                    confs[ri]))
                atomnum += 1
    print("END")
