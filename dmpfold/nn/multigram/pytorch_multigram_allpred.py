#!/usr/bin/env python

from __future__ import print_function

import sys
import os
import time

from math import sqrt, log

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from nndef_dist_gru2resnet import GRUResNet

# ############################## Main program ################################
# Everything else will be handled in our main program now. We could pull out
# more functions to better separate the code, but it wouldn't make it any
# easier to read.

# reweight MSA based on cutoff
def reweight(msa1hot, cutoff):
    id_min = msa1hot.shape[1] * cutoff
    id_mtx = torch.einsum('ikl,jkl->ij', msa1hot, msa1hot)
    id_mask = id_mtx > id_min
    w = 1. / id_mask.float().sum(dim=-1)
    return w


# shrunk covariance inversion
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


def main():

    device = torch.device("cpu")

    # Create neural network model (depending on first command line parameter)
    network = GRUResNet(512,128).eval().to(device)

    scriptdir = os.path.dirname(os.path.realpath(__file__))

    with open(sys.argv[1], 'r') as alnfile:
        aln = alnfile.read().splitlines()

    aa_trans = str.maketrans('ARNDCQEGHILKMFPSTWYVBJOUXZ-.', 'ABCDEFGHIJKLMNOPQRSTUUUUUUVV')

    nseqs = len(aln)
    length = len(aln[0])
    alnmat = (np.frombuffer(''.join(aln).translate(aa_trans).encode('latin-1'), dtype=np.uint8) - ord('A')).reshape(nseqs,length)

    inputs = torch.from_numpy(alnmat).type(torch.LongTensor).to(device)

    msa1hot = F.one_hot(torch.clamp(inputs, max=20), 21).float()
    w = reweight(msa1hot, cutoff=0.8)
    
    f2d_dca = fast_dca(msa1hot, w).float() if nseqs > 1 else torch.zeros((length, length, 442), device=device) 
    f2d_dca = f2d_dca.permute(2,0,1).unsqueeze(0)

    inputs2 = f2d_dca
    
    network.eval()
    with torch.no_grad():
        network.load_state_dict(torch.load(scriptdir + '/FINAL_fullmap_distcov_model1.pt', map_location=lambda storage, loc: storage))
        output = network(inputs, inputs2)
        network.load_state_dict(torch.load(scriptdir + '/FINAL_fullmap_distcov_model2.pt', map_location=lambda storage, loc: storage))
        output += network(inputs, inputs2)
        network.load_state_dict(torch.load(scriptdir + '/FINAL_fullmap_distcov_model3.pt', map_location=lambda storage, loc: storage))
        output += network(inputs, inputs2)
        network.load_state_dict(torch.load(scriptdir + '/FINAL_fullmap_distcov_model4.pt', map_location=lambda storage, loc: storage))
        output += network(inputs, inputs2)
        output /= 4
        output[0,0:2,:,:] = F.softmax(output[0,0:2,:,:], dim=0)
        output[0,2:36,:,:] = F.softmax(0.5 * (output[0,2:36,:,:] + output[0,2:36,:,:].transpose(-1,-2)), dim=0)
        output[0,36:70,:,:] = F.softmax(output[0,36:70,:,:], dim=0)
        output[0,70:104,:,:] = F.softmax(output[0,70:104,:,:], dim=0)

    with open(sys.argv[2] + ".hb", "w") as f:
        for wi in range(length):
            for wj in range(length):
                if abs(wi - wj) > 1:
                    probs = output.data[0,0:2,wi,wj]
                    print("{} {}".format(wi+1,wj+1), end='', file=f)
                    print(" 0 3.5 {}".format(torch.sum(probs[1:2])), end='', file=f)
                    print('', file=f)

    with open(sys.argv[2] + ".dist", "w") as f:
        for wi in range(0, length-5):
            for wj in range(wi+5, length):
                probs = output.data[0,2:36,wi,wj]
                print("{} {}".format(wi+1,wj+1), end='', file=f)
                for k in range(34):
                    print(" {}".format(probs[k]), end='', file=f)
                print('', file=f)

    with open(sys.argv[2] + ".phi", "w") as f:
        for wi in range(length):
            for wj in range(wi+1, length):
                probs = output.data[0,36:70,wi,wj]
                print("{} {}".format(wi+1,wj+1), end='', file=f)
                for k in range(34):
                    print(" {}".format(probs[k]), end='', file=f)
                print('', file=f)

    with open(sys.argv[2] + ".psi", "w") as f:
        for wi in range(length):
            for wj in range(wi+1, length):
                probs = output.data[0,70:104,wi,wj]
                print("{} {}".format(wi+1,wj+1), end='', file=f)
                for k in range(34):
                    print(" {}".format(probs[k]), end='', file=f)
                print('', file=f)

if __name__=="__main__":
    main()
