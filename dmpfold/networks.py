import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import pdist, squareform

from .nn.multigram.nndef_dist_gru2resnet import GRUResNet as GRUResNetInit
from .nn.multigram_iter.nndef_iterdist_gru2resnet import GRUResNet as GRUResNetIter

# Reweight MSA based on cutoff
def reweight(msa1hot, cutoff):
    id_min = msa1hot.shape[1] * cutoff
    id_mtx = torch.einsum('ikl,jkl->ij', msa1hot, msa1hot)
    id_mask = id_mtx > id_min
    w = 1. / id_mask.float().sum(dim=-1)
    return w

# Shrunk covariance inversion
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

# Extract a C-alpha distance matrix from a PDB file
def read_dmap(fp, length):
    coords = []
    with open(fp) as refpdbfile:
        for line in refpdbfile:
            if line[:4] == "ATOM" and line[12:16] == " CA ":
                # Split the line
                pdb_fields = [line[:6], line[6:11], line[12:16], line[17:20], line[21],
                                line[22:26], line[30:38], line[38:46], line[46:54]]
                coords.append(np.array([float(pdb_fields[6]), float(pdb_fields[7]),
                                        float(pdb_fields[8])]))
    assert length == len(coords)
    cv = np.asarray(coords)
    init_dmap = squareform(pdist(cv, "euclidean")).astype(np.float32).reshape(1,1, length, length)
    init_dmap = torch.from_numpy(init_dmap).type(torch.FloatTensor).contiguous()
    return init_dmap

# Use network to make predictions from alignment
def aln_to_predictions(aln_filepath, device="cpu"):
    # Create neural network model
    network = GRUResNetInit(512, 128).eval().to(device)

    scriptdir = os.path.dirname(os.path.realpath(__file__))

    with open(aln_filepath, 'r') as alnfile:
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
        model_dir = os.path.join(scriptdir, "nn", "multigram")
        network.load_state_dict(torch.load(os.path.join(model_dir, "FINAL_fullmap_distcov_model1.pt"),
                                map_location=device))
        output = network(inputs, inputs2)
        network.load_state_dict(torch.load(os.path.join(model_dir, "FINAL_fullmap_distcov_model2.pt"),
                                map_location=device))
        output += network(inputs, inputs2)
        network.load_state_dict(torch.load(os.path.join(model_dir, "FINAL_fullmap_distcov_model3.pt"),
                                map_location=device))
        output += network(inputs, inputs2)
        network.load_state_dict(torch.load(os.path.join(model_dir, "FINAL_fullmap_distcov_model4.pt"),
                                map_location=device))
        output += network(inputs, inputs2)
        output /= 4
        output[0,0:2,:,:] = F.softmax(output[0,0:2,:,:], dim=0)
        output[0,2:36,:,:] = F.softmax(0.5 * (output[0,2:36,:,:] + output[0,2:36,:,:].transpose(-1,-2)), dim=0)
        output[0,36:70,:,:] = F.softmax(output[0,36:70,:,:], dim=0)
        output[0,70:104,:,:] = F.softmax(output[0,70:104,:,:], dim=0)

    return output


# Use network and model structure to make predictions from alignment
def aln_to_predictions_iter(aln_filepath, ref_pdb_filepath, device="cpu"):
    # Create neural network model
    network = GRUResNetIter(512, 128).eval().to(device)

    scriptdir = os.path.dirname(os.path.realpath(__file__))

    with open(aln_filepath, 'r') as alnfile:
        aln = alnfile.read().splitlines()

    aa_trans = str.maketrans('ARNDCQEGHILKMFPSTWYVBJOUXZ-.', 'ABCDEFGHIJKLMNOPQRSTUUUUUUVV')

    nseqs = len(aln)
    length = len(aln[0])
    alnmat = (np.frombuffer(''.join(aln).translate(aa_trans).encode('latin-1'), dtype=np.uint8) - ord('A')).reshape(nseqs,length)

    init_dmap = read_dmap(ref_pdb_filepath, length).to(device)

    inputs = torch.from_numpy(alnmat).type(torch.LongTensor).to(device)

    msa1hot = F.one_hot(torch.clamp(inputs, max=20), 21).float()
    w = reweight(msa1hot, cutoff=0.8)

    f2d_dca = fast_dca(msa1hot, w).float() if nseqs > 1 else torch.zeros((length, length, 442), device=device)
    f2d_dca = f2d_dca.permute(2,0,1).unsqueeze(0)

    inputs2 = f2d_dca
    inputs2 = torch.cat((f2d_dca, init_dmap), dim=1)

    network.eval()
    with torch.no_grad():
        model_dir = os.path.join(scriptdir, "nn", "multigram_iter")
        network.load_state_dict(torch.load(os.path.join(model_dir, "FINAL_fullmap_distcov_model1.pt"),
                                map_location=device))
        output = network(inputs, inputs2)
        network.load_state_dict(torch.load(os.path.join(model_dir, "FINAL_fullmap_distcov_model2.pt"),
                                map_location=device))
        output += network(inputs, inputs2)
        network.load_state_dict(torch.load(os.path.join(model_dir, "FINAL_fullmap_distcov_model3.pt"),
                                map_location=device))
        output += network(inputs, inputs2)
        output /= 3
        output = torch.max(output, network(inputs, inputs2))
        output[0,0:2,:,:] = F.softmax(output[0,0:2,:,:], dim=0)
        output[0,2:36,:,:] = F.softmax(0.5 * (output[0,2:36,:,:] + output[0,2:36,:,:].transpose(-1,-2)), dim=0)
        output[0,36:70,:,:] = F.softmax(output[0,36:70,:,:], dim=0)
        output[0,70:104,:,:] = F.softmax(output[0,70:104,:,:], dim=0)

    return output

def write_predictions(output, prefix):
    length = output.size(2)

    with open(prefix + ".hb", "w") as f:
        for wi in range(length):
            for wj in range(length):
                if abs(wi - wj) > 1:
                    probs = output.data[0,0:2,wi,wj]
                    print("{} {}".format(wi+1,wj+1), end='', file=f)
                    print(" 0 3.5 {}".format(torch.sum(probs[1:2])), end='', file=f)
                    print('', file=f)

    with open(prefix + ".dist", "w") as f:
        for wi in range(0, length-5):
            for wj in range(wi+5, length):
                probs = output.data[0,2:36,wi,wj]
                print("{} {}".format(wi+1,wj+1), end='', file=f)
                for k in range(34):
                    print(" {}".format(probs[k]), end='', file=f)
                print('', file=f)

    with open(prefix + ".phi", "w") as f:
        for wi in range(length):
            for wj in range(wi+1, length):
                probs = output.data[0,36:70,wi,wj]
                print("{} {}".format(wi+1,wj+1), end='', file=f)
                for k in range(34):
                    print(" {}".format(probs[k]), end='', file=f)
                print('', file=f)

    with open(prefix + ".psi", "w") as f:
        for wi in range(length):
            for wj in range(wi+1, length):
                probs = output.data[0,70:104,wi,wj]
                print("{} {}".format(wi+1,wj+1), end='', file=f)
                for k in range(34):
                    print(" {}".format(probs[k]), end='', file=f)
                print('', file=f)
