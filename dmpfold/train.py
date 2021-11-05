# DMPfold2 end-to-end approach training script - D.T.Jones 2020

import sys
import os
import time
import random

from math import sqrt, log, exp

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

from network import GRUResNet

BATCH_SIZE = 32

# This should be adjusted according to size of GPU memory
MAXALNSZ = 300 * 1000

# Default crop length - try reducing e.g. to 300 if you are getting out of memory errors
DEF_CROPLEN = 350

# Max number of iterations - increase to 5 if >= 48 Gb GPU RAM
MAX_ITERATIONS = 3

# Restart from existing weights
RESTART_FLAG = True

# ################## Download and prepare the dataset ##################

def load_dataset():

    train_list = []
    validation_list = []
    tnum = 0

    with open('train_clust.lst', 'r') as targetfile:
        for line in targetfile:
            target_list = line.rstrip().split()

            print(' '.join(target_list))

            if tnum < 300:
                validation_list.append(target_list)
            else:
                train_list.append(target_list)

            tnum += 1

    return train_list, validation_list


# reweight MSA based on cutoff from https://github.com/lucidrains/tr-rosetta-pytorch
def reweight(msa1hot, cutoff):
    id_min = msa1hot.shape[1] * cutoff
    id_mtx = torch.einsum('ikl,jkl->ij', msa1hot, msa1hot)
    id_mask = id_mtx > id_min
    w = 1. / id_mask.float().sum(dim=-1)
    return w


# shrunk covariance inversion from https://github.com/lucidrains/tr-rosetta-pytorch
def fast_dca(msa1hot, weights, penalty = 4.5):
    device = msa1hot.device
    nr, nc, ns = msa1hot.shape
    x = msa1hot.view(nr, -1)
    num_points = weights.sum() - torch.sqrt(weights.mean())

    mean = (x * weights[:, None]).sum(dim=0, keepdims=True) / num_points
    x = (x - mean) * torch.sqrt(weights[:, None])

    cov = (x.t() @ x) / num_points
    cov_reg = cov + torch.eye(nc * ns, device=device) * penalty / torch.sqrt(weights.sum())

    try:
        inv_cov = torch.inverse(cov_reg)
    except:
        return None

    x1 = inv_cov.view(nc, ns, nc, ns)
    x2 = x1.transpose(1, 2).contiguous()
    features = x2.reshape(nc, nc, ns * ns)

    x3 = torch.sqrt((x1[:, :-1, :, :-1] ** 2).sum(dim=(1, 3))) * (1 - torch.eye(nc, device=device))
    apc = x3.sum(dim=0, keepdims=True) * x3.sum(dim=1, keepdims=True) / x3.sum()
    contacts = (x3 - apc) * (1 - torch.eye(nc, device=device))
    return torch.cat((features, contacts[:, :, None]), dim=2)


class DMPDataset(Dataset):

    def __init__(self, sample_list, augment=True):
        self.sample_list = sample_list
        self.augment = augment
        self.aanumdict = {'A':0, 'R':1, 'N':2, 'D':3, 'C':4, 'Q':5, 'E':6, 'G':7, 'H':8, 'I':9, 'L':10, 'K':11, 'M':12, 'F':13, 'P':14, 'S':15, 'T':16, 'W':17, 'Y':18, 'V':19, 'B':20, 'J':20, 'O':20, 'U':20, 'X':20, 'Z':20}
        self.aa_trans = str.maketrans('ARNDCQEGHILKMFPSTWYVBJOUXZ-.', 'ABCDEFGHIJKLMNOPQRSTUUUUUUVV')

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, tn):
        sample_list = self.sample_list

        if self.augment:
            targid = random.choice(sample_list[tn])
        else:
            targid = sample_list[tn][0]

        # Read coordinates from TDB file
        all_coords = []
        with open(os.path.join("tdb", targid + ".tdb"), "r") as tdbfile:
            for line in tdbfile:
                if (line[0] != '#'):
                    aanum = self.aanumdict.get(line[5], 21)
                    atoms = []
                    for i in range(5):
                        atoms.append((float(line[39+i*27:39+i*27+9]), float(line[39+i*27+9:39+i*27+18]), float(line[39+i*27+18:39+i*27+27])))
                    all_coords.append(np.asarray(atoms, dtype=np.float32))
        length = len(all_coords)

        targets = np.asarray(all_coords)

        # Read alignment from aln file
        with open(os.path.join("aln", targid + ".aln"), 'r') as alnfile:
            aln = alnfile.read().splitlines()

        nseqs = len(aln)
        alnmat = (np.frombuffer(''.join(aln).translate(self.aa_trans).encode('latin-1'), dtype=np.uint8) - ord('A')).reshape(nseqs,length)

        maxalignsz = MAXALNSZ

        if self.augment:
            # Initially crop ends according to terminal gaps in a random alignment row
            ns = random.randint(0, nseqs-1)
            aalocs = np.where(alnmat[ns] < 21)[0]
            alnmat = alnmat[:, aalocs[0]:aalocs[-1]+1]
            targets = targets[aalocs[0]:aalocs[-1]+1]
            length = alnmat.shape[1]
            # Crop if still over max length
            croplen = DEF_CROPLEN
            if length > croplen:
                lcut = random.randint(0, length-croplen)
                alnmat = alnmat[:,lcut:lcut+croplen]
                targets = targets[lcut:lcut+croplen]
                length = alnmat.shape[1]
            maxseqs = min(1000, maxalignsz // length)
            # Randomly sample raw alignment rows
            if nseqs > 1:
                p = (1 + int(exp(random.random() * log(nseqs-1)))) / nseqs
                rowmask = np.random.choice((False, True), size=nseqs, p=[1.0 - p, p])
                rowmask[0] = True
                alnmat = alnmat[rowmask,:]
                nseqs = alnmat.shape[0]
                if nseqs > maxseqs:
                    alnmat = alnmat[:maxseqs,:]
                    nseqs = maxseqs
        else:
            if nseqs > 1000:
                alnmat = alnmat[:1000,:]
                nseqs = 1000
            if length > 350:
                alnmat = alnmat[:,:350]
                targets = targets[:350]
                length = 350

        alnmatidx = torch.from_numpy(alnmat).type(torch.LongTensor).contiguous()

        # Calculate covariation data with gradients disabled
        with torch.no_grad():

            alnmatidx_gpu = alnmatidx.to(torch.device("cuda"))

            msa1hot = F.one_hot(torch.clamp(alnmatidx_gpu, max=20), 21).float()
            w = reweight(msa1hot, cutoff=0.8)

            f2d_dca = None
            if nseqs > 1:
                f2d_dca = fast_dca(msa1hot, w)
            if f2d_dca is None:
                f2d_dca = torch.zeros((1, 442, length, length), device=alnmatidx_gpu.device)
            else:
                f2d_dca = f2d_dca.permute(2,0,1).unsqueeze(0)

            inputs2 = f2d_dca

        targets = torch.from_numpy(targets)

        dmap = torch.zeros(1, 1, length, length) - 1
        inputs2 = torch.cat((inputs2.cpu(), dmap), dim=1)
        sample = (alnmatidx, inputs2, targets)

        return sample


# Trivial collate function
def my_collate(batch):
    return batch


# Returns residue TM-scores for two sets of coordinate c1 and c2 in shape (n_atoms, 3)
def tmscore(c1, c2):
    r1 = c1.transpose(0, 1)
    r2 = c2.transpose(0, 1)
    P = r1 - r1.mean(1).view(3, 1)
    Q = r2 - r2.mean(1).view(3, 1)
    cov = torch.matmul(P, Q.transpose(0, 1))
    try:
        U, S, Vh = torch.linalg.svd(cov)
        V = Vh.transpose(-2, -1).conj()
    except RuntimeError:
        return None
    d = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
        [0.0, 0.0, torch.det(torch.matmul(V, U.transpose(0, 1)))]], device=c1.device)
    rot = torch.matmul(torch.matmul(V, d), U.transpose(0, 1))
    rot_P = torch.matmul(rot, P)
    diffs = rot_P - Q
    d0sq = ((1.24 * diffs.size(1) / 5 - 15.0) ** (1.0/3.0) - 1.8) ** 2;
    tmscores = 1.0 / (1.0 + (diffs ** 2).sum(0) / d0sq)
    return tmscores


# Starts here

def main(num_epochs=1000):

    device = torch.device("cuda")

    # Create neural network model1
    network = GRUResNet(512,128).to(device)

    # Load the dataset
    print("Loading data...")
    train_list, validation_list = load_dataset()

    ntrain = len(train_list)
    nvalidation = len(validation_list)

    train_err_min = 1e32
    val_err_min = 1e32

    # Load current model snapshot
    if RESTART_FLAG:
        pretrained_dict = torch.load('fullmap_e2e_model_train.pt', map_location=lambda storage, loc: storage)
        model_dict = network.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict) and (model_dict[k].shape == pretrained_dict[k].shape)}
        network.load_state_dict(pretrained_dict, strict=False)
        max_lr = 1e-4
    else:
        max_lr = 3e-4

    optimizer = torch.optim.Adam(network.parameters(), lr=max_lr)

    scaler = torch.cuda.amp.GradScaler()

    try:
        checkpoint = torch.load('checkpoint.pt')
        optimizer.load_state_dict(checkpoint['optimizer.state_dict'])
        scaler.load_state_dict(checkpoint['scaler.state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        val_err_min = checkpoint['val_err_min']
        train_err_min = checkpoint['train_err_min']
        print("Checkpoint file loaded.")
    except:
        start_epoch = 0
        print("WARNING: No checkpoint file found!")
    
    dmp_train_data = DMPDataset(train_list, augment=True)
    dmp_val_data = DMPDataset(validation_list, augment=False)

    data_loader = DataLoader(dataset=dmp_train_data, 
                             batch_size=BATCH_SIZE,
                             shuffle=True,
                             drop_last=True,
                             num_workers=1,
                             pin_memory=True,
                             collate_fn=my_collate)

    val_data_loader = DataLoader(dataset=dmp_val_data, 
                             batch_size=4,
                             shuffle=False,
                             drop_last=False,
                             num_workers=1,
                             pin_memory=True,
                             collate_fn=my_collate)


    def run_sample(sample, batch_len, nloops):
        inputs = sample[0].to(device, non_blocking=True)
        inputs2 = sample[1].to(device, non_blocking=True)
        targets = sample[2].to(device, non_blocking=True)

        nres = inputs2.size(-1)

        # Reduce this probability towards zero for later training epochs
        if random.random() < 0.5:
            # Sometimes seed with noised target structure
            coords = targets[:,1:2,:]
            coords += 0.5 * torch.randn_like(coords)
            dmap = (coords - coords.transpose(0,1)).pow(2).sum(dim=2).sqrt()
            inputs2[0, -1] = dmap

        with torch.cuda.amp.autocast():
            coords, confs = network(inputs, inputs2, nloops=nloops, refine_steps=100)

            coords = coords.view(nres,5,3)
            confs = confs[0,:]

            nres = coords.size(0)

        tmscores = tmscore(targets.view(nres*5, 3), coords.view(nres*5, 3))
        if tmscores is None:
            return None

        with torch.cuda.amp.autocast():
            coord_loss = (1 - tmscores).mean()

            conf_loss = (confs - tmscores.detach()[1::5]).abs().mean()

            # Loss for correct C-alpha stereochemistry
            dsqmap = coords[:,1:2,:]
            dsqmap = (dsqmap - dsqmap.transpose(0,1)).pow(2).sum(dim=2)
            steric_loss = torch.triu(F.relu(9.0 - dsqmap), diagonal=2).sum()
            steric_loss = torch.tanh(steric_loss + (torch.clip(torch.diag(dsqmap, diagonal=1).sqrt(), min=1e-8) - 3.78).pow(2).sum() / 64.0)

            loss = coord_loss + conf_loss + 0.02 * steric_loss

        if loss.requires_grad:
            scaler.scale(loss / batch_len).backward()

        return loss.item()

    # Finally, launch the training loop.
    print("Starting training...")

    for epoch in range(start_epoch, start_epoch+num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_samples = 0
        start_time = time.time()

        sys.stdout.flush()
        network.train()
        random.seed()

        for sample_batch in data_loader:
            print("hi")
            network.zero_grad(set_to_none=True)
            random.shuffle(sample_batch)
            batch_len = len(sample_batch)
            for sample in sample_batch:
                torch.cuda.empty_cache()
                # Run random number of iterations - note max. number can be increased with more GPU RAM
                # 5 loops will probably require 48 Gb GPU RAM for training
                loss = run_sample(sample, batch_len, random.randint(0, MAX_ITERATIONS))
                if loss is not None:
                    train_err += loss
                    train_samples += 1
            scaler.step(optimizer)
            scaler.update()

        # And a full pass over the validation set:
        val_err = 0.0
        val_samples = 0

        network.eval()

        # Fix random seed for validation
        random.seed(1)
        
        with torch.no_grad():
            for sample_batch in val_data_loader:
                for sample in sample_batch:
                    torch.cuda.empty_cache()
                    loss = run_sample(sample, 1, 2)
                    if loss is not None:
                        val_err += loss
                        val_samples += 1

        print(train_samples, val_samples)

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, start_epoch + num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_samples))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_samples))
            
        if val_err < val_err_min:
            val_err_min = val_err
            torch.save(network.state_dict(), 'fullmap_e2e_model.pt')
            print("Saving model...")

        if train_err < train_err_min:
            train_err_min = train_err
            torch.save(network.state_dict(), 'fullmap_e2e_model_train.pt')
            print("Saving best training error model...")

        torch.save({
            'epoch': epoch,
            'optimizer.state_dict': optimizer.state_dict(),
            'scaler.state_dict': scaler.state_dict(),
            'val_err_min': val_err_min,
            'train_err_min': train_err_min
            }, 'checkpoint.pt')

if __name__ == "__main__":
    main()
