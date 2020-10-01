import sys
import os
from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import pdist, squareform

NUM_CHANNELS = 442

class Maxout2d(nn.Module):
    def __init__(self, in_channels, out_channels, pool_size, kernel_size=1, dilation=1, block=0):
        super(Maxout2d, self).__init__()
        self.in_channels, self.out_channels, self.pool_size = in_channels, out_channels, pool_size
        self.lin = nn.Conv2d(in_channels=in_channels, out_channels=out_channels * pool_size, kernel_size=kernel_size, dilation=dilation, padding=dilation*(kernel_size-1)//2)
        self.norm = nn.InstanceNorm2d(out_channels, affine=True)
        if block > 0:
            nn.init.xavier_uniform_(self.lin.weight, gain=1.0 / sqrt(block))
        else:
            nn.init.xavier_uniform_(self.lin.weight, gain=1.0)

    def forward(self, inputs):
        x = self.lin(inputs)

        N, C, H, W = x.size()

        x = x.view(N, C//self.pool_size, self.pool_size, H, W)
        x = x.max(dim=2)[0]
        x = self.norm(x)

        return x

class CSE(nn.Module):
    def __init__(self, width, reduction=16):
        super(CSE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(width, width // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(width // reduction, width, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        N, C, _, _ = x.size()
        y = self.avg_pool(x).view(N, C)
        y = self.fc(y).view(N, C, 1, 1)

        return x * y.expand_as(x)

class SSE(nn.Module):
    def __init__(self, in_ch):
        super(SSE, self).__init__()

        self.conv = nn.Conv2d(in_ch, 1, kernel_size=1, stride=1)

    def forward(self, x):
        y = self.conv(x)
        y = torch.sigmoid(y)

        return x * y

class SCSE(nn.Module):
    def __init__(self, in_ch, r):
        super(SCSE, self).__init__()

        self.cSE = CSE(in_ch, r)
        self.sSE = SSE(in_ch)

    def forward(self, x):
        cSE = self.cSE(x)
        sSE = self.sSE(x)

        return cSE + sSE

# ResNet Module
class ResNet_Block(nn.Module):
    def __init__(self,width,fsize,dilv,nblock):
        super(ResNet_Block, self).__init__()
        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout2d(p=0.2)
        self.layer1 = Maxout2d(in_channels=width, out_channels=width, pool_size=4, kernel_size=fsize, dilation=dilv, block=nblock)
        self.scSE = SCSE(width, 16)

    def forward(self, x):
        residual = x
        out = self.dropout1(x)
        out = self.dropout2(out)
        out = self.layer1(out)
        out = self.scSE(out)
        out = out + residual

        return out

# RNNResNet Module
class GRUResNetInit(nn.Module):
    def __init__(self,width,cwidth):
        super(GRUResNetInit, self).__init__()

        self.width = width
        self.cwidth = cwidth

        self.embed = nn.Embedding.from_pretrained(torch.eye(22), freeze=True)
        self.vgru = nn.GRU(22, width, batch_first=False, num_layers=2, bidirectional=False)
        self.hgru = nn.GRU(width, width//2, batch_first=False, num_layers=2, dropout=0.1, bidirectional=True)

        layers = []

        layer = Maxout2d(in_channels=NUM_CHANNELS+width, out_channels=cwidth, pool_size=3)

        layers.append(layer)

        nblock = 1

        for rep in range(16):
            for fsize,dilv in [(5,1)]:
                if fsize > 0:
                    layer = ResNet_Block(cwidth, fsize, dilv, nblock)
                    layers.append(layer)
                    nblock += 1

        layer = nn.Conv2d(in_channels=cwidth, out_channels=2+3*34, kernel_size=1)
        layer.weight.data.zero_()
        layer.bias.data.zero_()
        layers.append(layer)

        self.resnet = nn.Sequential(*layers)

    def forward(self, x, x2):
        nseqs = x.size()[0]
        nres = x.size()[1]

        x = self.embed(x)
        xx = torch.empty((0, self.width), device=x.device)
        for i in range(0, nres, 50):
            xx = torch.cat((xx, self.vgru(x[:,i:min(nres, i+50)])[0][-1,:,:]))
        x = self.hgru(xx.unsqueeze(1))[0]

        mat1d = x.permute(1,2,0)
        mat2d = mat1d.unsqueeze(2).expand(1, self.width, nres, nres)
        mat2d_T = mat1d.unsqueeze(3).expand(1, self.width, nres, nres)
        x = mat2d * mat2d_T

        x = torch.cat((x, x2), dim=1)

        out = self.resnet(x)

        return out

# RNNResNet Module
class GRUResNetIter(nn.Module):
    def __init__(self,width,cwidth):
        super(GRUResNetIter, self).__init__()

        self.width = width
        self.cwidth = cwidth

        self.embed = nn.Embedding.from_pretrained(torch.eye(22), freeze=True)
        self.vgru = nn.GRU(22, width, batch_first=False, num_layers=2, bidirectional=False)
        self.hgru = nn.GRU(width, width//2, batch_first=False, num_layers=2, dropout=0.1, bidirectional=True)

        layers = []

        layer = Maxout2d(in_channels=NUM_CHANNELS+1+width, out_channels=cwidth, pool_size=3)

        layers.append(layer)

        nblock = 1

        for rep in range(16):
            for fsize,dilv in [(5,1)]:
                if fsize > 0:
                    layer = ResNet_Block(cwidth, fsize, dilv, nblock)
                    layers.append(layer)
                    nblock += 1

        layer = nn.Conv2d(in_channels=cwidth, out_channels=2+3*34, kernel_size=1)
        layer.weight.data.zero_()
        layer.bias.data.zero_()
        layers.append(layer)

        self.resnet = nn.Sequential(*layers)

    def forward(self, x, x2):
        nseqs = x.size()[0]
        nres = x.size()[1]

        x = self.embed(x)
        x = self.vgru(x)[0]
        x = self.hgru(x[-1,:,:].unsqueeze(1))[0]

        mat1d = x.permute(1,2,0)
        mat2d = mat1d.unsqueeze(2).expand(1, self.width, nres, nres)
        mat2d_T = mat1d.unsqueeze(3).expand(1, self.width, nres, nres)
        x = mat2d * mat2d_T

        x = torch.cat((x, x2), dim=1)

        out = self.resnet(x)

        return out

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
