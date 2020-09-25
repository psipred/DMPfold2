import sys
import os
import shutil
from math import pi, radians
from random import random
from itertools import count
from datetime import datetime
from contextlib import suppress, redirect_stdout

import torch
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import pdist, squareform

from modeller import *
from modeller.automodel import *
from modeller.scripts import complete_pdb

from PeptideBuilder import PeptideBuilder

from .nn.multigram.nndef_dist_gru2resnet import GRUResNet as GRUResNetInit
from .nn.multigram_iter.nndef_iterdist_gru2resnet import GRUResNet as GRUResNetIter

dev    = "cuda" if torch.cuda.is_available() else "cpu" # Force-directed folding device
dev_nn = "cpu" # Neural network inference device

n_iters       = 3      # Number of iterations
n_trajs       = 20     # Number of folding trajectories
n_steps       = 20_000 # Steps per folding simulation
gauss_sigma   = 1.25   # Width of Gaussians
k_vdw         = 100.0  # Force constant for vdw exclusion
n_min_steps   = 5000   # Minimisation steps to finish simulation
min_speed     = 0.002  # Minimisation speed
k_dih         = 1.0    # Force constant for dihedrals
omega_dev     = 0.2    # Omega dihedral deviation
k_hb_dist     = 5.0    # Force constant for hydrogen bond distances
hb_cutoff_gen = 6.0    # Cutoff for general hydrogen bond potential
k_hb_gen      = 0.0    # Force constant for general hydrogen bond distance potential
k_ang_hb_gen  = 0.0    # Force constant for general hydrogen bond angle potential
hb_res_sep    = 10     # Minimum separation for predicted hydrogen bonds

keep_tempdir = False # Whether to keep temporary directory with intermediate files
record_n = 4_000 # Interval for logging trajectory, set to 0 to not print
debug_n = 0 # Print forces every this many steps, set to 0 to disable
integrator = "no_vel" # vel is velocity Verlet, no_vel is velocity-free Verlet, min is minimisation
final_from_all_iters = False # Select the final model from all iterations
coil_level = 50.0 # Degree of curvature in starting conformations
starting_vel = 0.05
timestep = 0.01
temperature_start = 1.0 # Pseudo-temperature at start
temperature_end = 0.1 # Pseudo-temperature at end, linear annealing from start to end
thermostat_const = 10.0 # Degree of coupling to thermostat
pairwise_exponent_min = 1.0
weight_violate_min = 3.0 # Violating a min bound is this many times worse than a max bound

k_cb_dist = 150.0
k_cov = 100.0 # Force constant for covalent bonds
k_ang = k_cov # Force constant for bond angles
vdw_dist = 3.0 # Distance for vdw exclusion
pairwise_exponent_max = 0.75 # 1 for linear force scaling (harmonic potential), 0.5 for sub-linear scaling
pairwise_force_cutoff = 40.0 # Forces no longer increase above this distance
hb_prob_init, hb_prob_iter = 0.31, 0.17
phi_prob_init, phi_prob_iter = 0.76, 0.66
psi_prob_init, psi_prob_iter = 0.16, 0.71

n_bins = 34
dist_bin_centres = torch.tensor([3.75 + 0.5 * i for i in range(n_bins)], device=dev)
sigmas = torch.tensor([gauss_sigma] * n_bins, device=dev)

one_to_three_aas = {"C": "CYS", "D": "ASP", "S": "SER", "Q": "GLN", "K": "LYS",
                    "I": "ILE", "P": "PRO", "T": "THR", "F": "PHE", "N": "ASN",
                    "G": "GLY", "H": "HIS", "L": "LEU", "R": "ARG", "W": "TRP",
                    "A": "ALA", "V": "VAL", "E": "GLU", "Y": "TYR", "M": "MET"}

atoms = ("N", "CA", "C", "O", "CB", "H")
elements = {"N": "N", "CA": "C", "C": "C", "O": "O", "CB": "C", "H": "H"}

true_cov = {"N-CA": 1.46, "CA-C": 1.53, "C-N": 1.33, "CA-CB": 1.54, "N-H": 1.01, "C-O": 1.24}
true_ang = {
        "N-CA-C": 1.94, "CA-C-N": 2.02, "C-N-CA": 2.13, "N-CA-CB": 1.93, "C-CA-CB": 1.91,
        "CA-N-H": 1.99, "C-N-H" : 2.15, "N-C-O" : 2.18, "CA-C-O" : 2.11, "O-H-N"  : pi  ,
        "C-O-H" : pi  ,
}

class ForceFolder(torch.nn.Module):
    def __init__(self, sequence, n_trajs, distance_preds, hb_preds, dihedral_preds):
        super(ForceFolder, self).__init__()
        self.sequence = sequence
        self.n_res = len(self.sequence)
        self.n_trajs = n_trajs
        self.gaussian_weights = distance_preds
        self.hb_dists_min, self.hb_dists_max = hb_preds["min"], hb_preds["max"]
        self.phis_mean, self.phis_dev = dihedral_preds["phis_mean"], dihedral_preds["phis_dev"]
        self.psis_mean, self.psis_dev = dihedral_preds["psis_mean"], dihedral_preds["psis_dev"]
        self.omegas_mean = torch.ones(self.n_res - 1, device=dev) * pi
        self.omegas_dev = torch.ones(self.n_res - 1, device=dev) * omega_dev

        # Generate coiled starting conformations using PeptideBuilder
        self.coords = {}
        structures = [PeptideBuilder.initialize_res("A") for _ in range(self.n_trajs)]
        phi_shifts = torch.ones(self.n_trajs, device=dev) * -120
        psi_shifts = torch.ones(self.n_trajs, device=dev) * 140
        for i in range(self.n_res - 1):
            phi_shifts += torch.randn(self.n_trajs, device=dev) * coil_level
            psi_shifts += torch.randn(self.n_trajs, device=dev) * coil_level
            for j in range(self.n_trajs):
                structures[j] = PeptideBuilder.add_residue(structures[j], "A",
                                                        phi_shifts[j].item(), psi_shifts[j].item())
        for atom in atoms:
            self.coords[atom] = torch.zeros(self.n_trajs, self.n_res, 3, device=dev)
            for j in range(self.n_trajs):
                chain = structures[j][0]["A"]
                # PeptideBuilder does not place amide hydrogens so we add them manually
                if atom == "H":
                    for i in range(self.n_res):
                        vec_ca_n = chain[i + 1]["N"].coord - chain[i + 1]["CA"].coord
                        if i == 0:
                            vec_n_h = torch.tensor(vec_ca_n, device=dev)
                        else:
                            vec_c_n = chain[i + 1]["N"].coord - chain[i]["C"].coord
                            vec_n_h = torch.tensor(vec_c_n + vec_ca_n, device=dev)
                        self.coords["H"][j, i] = torch.tensor(chain[i + 1]["N"].coord,
                                                device=dev) + F.normalize(vec_n_h, dim=0) * true_cov["N-H"]
                else:
                    self.coords[atom][j] = torch.tensor(
                        [list(chain[i + 1][atom].coord) for i in range(self.n_res)], device=dev)

        # Find likely minimum distogram energy
        # Since we use the maximum probability bin this might not technically be the minimum
        min_bins_flat = self.gaussian_weights.argmin(dim=2).view(self.n_res * self.n_res)
        min_dists = dist_bin_centres.index_select(0, min_bins_flat).view(self.n_res, self.n_res)
        dists_from_means = min_dists.unsqueeze(2) - dist_bin_centres.view(1, 1, n_bins)
        min_cb_energies = self.gaussian_weights * torch.exp(-0.5 * (dists_from_means / sigmas.view(
                                                1, 1, n_bins)).pow(2)) / sigmas.view(1, 1, n_bins)
        self.min_cb_energy = min_cb_energies.sum()

    def fold(self, n_steps):
        # Run folding simulations for a given number of steps
        # See https://arxiv.org/pdf/1401.1181.pdf for derivation of forces
        if integrator == "vel":
            vels = {}
            accs_last = {}
            for atom in atoms:
                vels[atom] = torch.randn(self.n_trajs, self.n_res, 3, device=dev) * starting_vel
                accs_last[atom] = torch.zeros(self.n_trajs, self.n_res, 3, device=dev)
        elif integrator == "no_vel":
            coords_last = {}
            for atom in atoms:
                coords_last[atom] = self.coords[atom].clone() + torch.randn(
                                self.n_trajs, self.n_res, 3, device=dev) * starting_vel * timestep

        diag_mask = torch.eye(self.n_res, device=dev)
        for offset in (1, 2):
            diag_mask += torch.diag_embed(torch.ones(self.n_res - offset, device=dev), offset= offset)
            diag_mask += torch.diag_embed(torch.ones(self.n_res - offset, device=dev), offset=-offset)
        diag_mask = (1.0 - diag_mask).unsqueeze(0).expand(self.n_trajs, -1, -1)

        sigmas_rep = sigmas.view(1, 1, 1, n_bins)

        for step_n in range(n_steps + n_min_steps):
            if integrator == "vel" and step_n < n_steps:
                for atom in atoms:
                    self.coords[atom] += vels[atom] * timestep + 0.5 * accs_last[atom] * timestep ** 2

            accs = {}
            for atom in atoms:
                accs[atom] = torch.zeros(self.n_trajs, self.n_res, 3, device=dev)

            # Forces due to distogram CB-CB potentials
            crep_cb = self.coords["CB"].unsqueeze(1).expand(-1, self.n_res, -1, -1)
            diffs_cb = crep_cb - crep_cb.transpose(1, 2)
            dists_cb = diffs_cb.norm(dim=3).clamp(min=0.1, max=pairwise_force_cutoff)
            norm_diffs_cb = diffs_cb / dists_cb.unsqueeze(3)
            dists_from_means = dists_cb.unsqueeze(3) - dist_bin_centres.view(1, 1, 1, n_bins)
            cb_energies = self.gaussian_weights.unsqueeze(0) * torch.exp(
                                    -0.5 * (dists_from_means / sigmas_rep).pow(2)) / sigmas_rep
            forces = ((-dists_from_means / sigmas_rep.pow(2)) * cb_energies).sum(dim=3)
            pair_accs = -forces.unsqueeze(3) * k_cb_dist * norm_diffs_cb
            accs["CB"] += pair_accs.sum(dim=1)
            if debug_n > 0 and step_n % debug_n == 0:
                print(f"Cb{accs['CB'].abs().mean().item():9.3f}", end="  ")

            # Forces due to hydrogen bond distances
            # Predicted hydrogen bonds have a long range force, all hydrogen bonds have a short
            #   range force to promote secondary structure formation
            crep_h = self.coords["H"].unsqueeze(1).expand(-1, self.n_res, -1, -1)
            crep_o = self.coords["O"].unsqueeze(1).expand(-1, self.n_res, -1, -1)
            diffs_ho = crep_o - crep_h.transpose(1, 2)
            dists_ho = diffs_ho.norm(dim=3).clamp(min=0.1, max=pairwise_force_cutoff)
            norm_diffs_ho = diffs_ho / dists_ho.unsqueeze(3)
            violate_min_hb = (self.hb_dists_min - dists_ho).clamp(min=0.0)
            violate_max_hb = (dists_ho - self.hb_dists_max).clamp(min=0.0)
            close_hbonds = (dists_ho < hb_cutoff_gen).to(torch.float) * diag_mask
            violate_max_hb_gen = close_hbonds * (dists_ho - hb_cutoff_gen).clamp(min=0.0)
            forces_min = weight_violate_min * k_hb_dist * violate_min_hb ** pairwise_exponent_min
            forces_max = k_hb_dist * violate_max_hb ** pairwise_exponent_max
            forces_max += k_hb_gen * violate_max_hb_gen ** pairwise_exponent_max
            pair_accs = forces_min.unsqueeze(3) * norm_diffs_ho - forces_max.unsqueeze(3) * norm_diffs_ho
            accs["H"] -= pair_accs.sum(dim=2)
            accs["O"] += pair_accs.sum(dim=1)
            if debug_n > 0 and step_n % debug_n == 0:
                print(f"hb{pair_accs.sum(dim=2).abs().mean().item():9.3f}", end="  ")

            # Forces due to hydrogen bond angles
            # All close hydrogen bonds have an angle constraint
            crep_n = self.coords["N"].unsqueeze(1).expand(-1, self.n_res, -1, -1)
            diffs_hn = (crep_n - crep_h).transpose(1, 2)
            ba_norms = diffs_ho.norm(dim=3)
            bc_norms = diffs_hn.norm(dim=3)
            angs = torch.acos(((diffs_ho * diffs_hn).sum(dim=3) / (ba_norms * bc_norms).clamp(
                                min=0.1, max=100.0)).clamp(min=-1.0, max=1.0))
            violate = angs - true_ang["O-H-N"]
            forces = k_ang_hb_gen * close_hbonds * violate
            cross_ba_bc = torch.cross(diffs_ho, diffs_hn, dim=3)
            fa_vec = torch.cross(-diffs_ho, cross_ba_bc, dim=3)
            fc_vec = torch.cross( diffs_hn, cross_ba_bc, dim=3)
            # Don't divide by length
            fa = forces.unsqueeze(3) * fa_vec / fa_vec.norm(dim=3).unsqueeze(3).clamp(min=0.1)
            fc = forces.unsqueeze(3) * fc_vec / fc_vec.norm(dim=3).unsqueeze(3).clamp(min=0.1)
            fb = -fa - fc
            accs["O"] += fa.sum(dim=1)
            accs["H"] += fb.sum(dim=2)
            accs["N"] += fc.sum(dim=2)
            if debug_n > 0 and step_n % debug_n == 0:
                print(f"hba{fb.sum(dim=2).abs().mean().item():9.3f}", end="  ")

            crep_c = self.coords["C"].unsqueeze(1).expand(-1, self.n_res, -1, -1)
            diffs_oc = (crep_c - crep_o).transpose(1, 2)
            ba_norms = diffs_oc.norm(dim=3)
            bc_norms = diffs_ho.norm(dim=3)
            angs = torch.acos(((diffs_oc * -diffs_ho).sum(dim=3) / (ba_norms * bc_norms).clamp(
                                min=0.1, max=100.0)).clamp(min=-1.0, max=1.0))
            violate = angs - true_ang["C-O-H"]
            forces = k_ang_hb_gen * close_hbonds * violate
            cross_ba_bc = torch.cross(diffs_oc, -diffs_ho, dim=3)
            fa_vec = torch.cross(-diffs_oc, cross_ba_bc, dim=3)
            fc_vec = torch.cross(-diffs_ho, cross_ba_bc, dim=3)
            fa = forces.unsqueeze(3) * fa_vec / fa_vec.norm(dim=3).unsqueeze(3).clamp(min=0.1)
            fc = forces.unsqueeze(3) * fc_vec / fc_vec.norm(dim=3).unsqueeze(3).clamp(min=0.1)
            fb = -fa - fc
            accs["C"] += fa.sum(dim=1)
            accs["O"] += fb.sum(dim=1)
            accs["H"] += fc.sum(dim=2)

            # Forces due to vdw exclusion
            # Only check for pairs of each atom type
            for atom in atoms:
                crep = self.coords[atom].unsqueeze(1).expand(-1, self.n_res, -1, -1)
                diffs = crep - crep.transpose(1, 2)
                dists = diffs.norm(dim=3).clamp(min=0.01, max=10.0)
                norm_diffs = diffs / dists.unsqueeze(3)
                violate = (dists < vdw_dist).to(torch.float) * (vdw_dist - dists)
                forces = k_vdw * violate
                pair_accs = forces.unsqueeze(3) * norm_diffs
                accs[atom] += pair_accs.sum(dim=1)
                if debug_n > 0 and step_n % debug_n == 0 and atom == "N":
                    print(f"vdw{pair_accs.sum(dim=1).abs().mean().item():8.3f}", end="  ")

            # Forces due to covalent bonds
            # across_res is whether atom_2 is in the next residue
            for atom_1, atom_2, across_res in ( ("N", "CA", 0), ("CA", "C" , 0),
                                                ("C", "N" , 1), ("CA", "CB", 0),
                                                ("N", "H" , 0), ("C" , "O" , 0)):
                if across_res == 0:
                    diffs = self.coords[atom_2] - self.coords[atom_1]
                elif across_res == 1:
                    diffs = self.coords[atom_2][:, 1:] - self.coords[atom_1][:, :-1]
                dists = diffs.norm(dim=2).clamp(min=0.1)
                norm_diffs = diffs / dists.unsqueeze(2)
                key = f"{atom_1}-{atom_2}"
                violate = dists - true_cov[key]
                forces = k_cov * violate
                accs_cov = forces.unsqueeze(2) * norm_diffs
                if across_res == 0:
                    accs[atom_1] += accs_cov
                    accs[atom_2] -= accs_cov
                elif across_res == 1:
                    accs[atom_1][:, :-1] += accs_cov
                    accs[atom_2][:, 1: ] -= accs_cov
                if debug_n > 0 and step_n % debug_n == 0 and key == "N-CA":
                    print(f"cov{accs_cov.abs().mean().item():8.3f}", end="  ")

            # Forces due to bond angles
            # across_res is the number of atoms in the next residue, starting from atom_3
            for atom_1, atom_2, atom_3, across_res in (("N", "CA", "C" , 0), ("CA", "C", "N" , 1),
                                ("C" , "N", "CA", 2),  ("N", "CA", "CB", 0), ("C", "CA", "CB", 0),
                                ("CA", "N", "H" , 0),  ("C", "N" , "H" , 2), ("N", "C" , "O" , 1),
                                ("CA", "C", "O" , 0)):
                if across_res == 0:
                    ba = self.coords[atom_1] - self.coords[atom_2]
                    bc = self.coords[atom_3] - self.coords[atom_2]
                elif across_res == 1:
                    ba = self.coords[atom_1][:, :-1] - self.coords[atom_2][:, :-1]
                    bc = self.coords[atom_3][:, 1: ] - self.coords[atom_2][:, :-1]
                elif across_res == 2:
                    ba = self.coords[atom_1][:, :-1] - self.coords[atom_2][:, 1: ]
                    bc = self.coords[atom_3][:, 1: ] - self.coords[atom_2][:, 1: ]
                ba_norms = ba.norm(dim=2)
                bc_norms = bc.norm(dim=2)
                angs = torch.acos(((ba * bc).sum(dim=2) / (ba_norms * bc_norms).clamp(min=0.1, max=100.0)).clamp(min=-1.0, max=1.0))
                key = f"{atom_1}-{atom_2}-{atom_3}"
                violate = angs - true_ang[key]
                forces = k_ang * violate
                cross_ba_bc = torch.cross(ba, bc, dim=2)
                fa_vec = torch.cross(-ba, cross_ba_bc, dim=2)
                fc_vec = torch.cross( bc, cross_ba_bc, dim=2)
                fa = (forces / ba_norms).unsqueeze(2) * fa_vec / fa_vec.norm(dim=2).unsqueeze(2).clamp(min=0.1)
                fc = (forces / bc_norms).unsqueeze(2) * fc_vec / fc_vec.norm(dim=2).unsqueeze(2).clamp(min=0.1)
                fb = -fa - fc
                if across_res == 0:
                    accs[atom_1] += fa
                    accs[atom_2] += fb
                    accs[atom_3] += fc
                if across_res == 1:
                    accs[atom_1][:, :-1] += fa
                    accs[atom_2][:, :-1] += fb
                    accs[atom_3][:, 1: ] += fc
                if across_res == 2:
                    accs[atom_1][:, :-1] += fa
                    accs[atom_2][:, 1: ] += fb
                    accs[atom_3][:, 1: ] += fc
                if debug_n > 0 and step_n % debug_n == 0 and key == "N-CA-C":
                    print(f"ang{fa.abs().mean().item():8.3f}", end="  ")

            # Forces due to dihedral angles
            # across_res is the number of atoms in the next residue, starting from atom_4
            for atom_1, atom_2, atom_3, atom_4, across_res, dmean, ddev in (
                                ("C" , "N" , "CA", "C" , 3, self.phis_mean  , self.phis_dev  ),
                                ("N" , "CA", "C" , "N" , 1, self.psis_mean  , self.psis_dev  ),
                                ("CA", "C" , "N" , "CA", 2, self.omegas_mean, self.omegas_dev)):
                if across_res == 1:
                    ab = self.coords[atom_2][:, :-1] - self.coords[atom_1][:, :-1]
                    bc = self.coords[atom_3][:, :-1] - self.coords[atom_2][:, :-1]
                    cd = self.coords[atom_4][:, 1: ] - self.coords[atom_3][:, :-1]
                elif across_res == 2:
                    ab = self.coords[atom_2][:, :-1] - self.coords[atom_1][:, :-1]
                    bc = self.coords[atom_3][:, 1: ] - self.coords[atom_2][:, :-1]
                    cd = self.coords[atom_4][:, 1: ] - self.coords[atom_3][:, 1: ]
                elif across_res == 3:
                    ab = self.coords[atom_2][:, 1: ] - self.coords[atom_1][:, :-1]
                    bc = self.coords[atom_3][:, 1: ] - self.coords[atom_2][:, 1: ]
                    cd = self.coords[atom_4][:, 1: ] - self.coords[atom_3][:, 1: ]
                cross_ab_bc = torch.cross(ab, bc, dim=2)
                cross_bc_cd = torch.cross(bc, cd, dim=2)
                bc_norms = bc.norm(dim=2).unsqueeze(2)
                dihs = torch.atan2(
                    torch.sum(torch.cross(cross_ab_bc, cross_bc_cd, dim=2) * bc / bc_norms, dim=2),
                    torch.sum(cross_ab_bc * cross_bc_cd, dim=2)
                )
                # Get angular difference, shift, use remainder, shift back, normalise
                # Runs from -1 to 1 with 0 being no violation
                dih_violate = (((dmean - dihs + pi) % (2 * pi)) - pi) / pi
                forces = k_dih * dih_violate / ddev
                fa = (forces / ab.norm(dim=2)).unsqueeze(2) * F.normalize(-cross_ab_bc, dim=2)
                fd = (forces / cd.norm(dim=2)).unsqueeze(2) * F.normalize( cross_bc_cd, dim=2)
                # Forces on the middle atoms have to keep the sum of torques null
                # Forces taken from http://www.softberry.com/freedownloadhelp/moldyn/description.html
                fb = ((ab * -bc) / (bc_norms ** 2) - 1) * fa - ((cd * -bc) / (bc_norms ** 2)) * fd
                fc = -fa - fb - fd
                if across_res == 1:
                    accs[atom_1][:, :-1] += fa
                    accs[atom_2][:, :-1] += fb
                    accs[atom_3][:, :-1] += fc
                    accs[atom_4][:, 1: ] += fd
                elif across_res == 2:
                    accs[atom_1][:, :-1] += fa
                    accs[atom_2][:, :-1] += fb
                    accs[atom_3][:, 1: ] += fc
                    accs[atom_4][:, 1: ] += fd
                elif across_res == 3:
                    accs[atom_1][:, :-1] += fa
                    accs[atom_2][:, 1: ] += fb
                    accs[atom_3][:, 1: ] += fc
                    accs[atom_4][:, 1: ] += fd
                if debug_n > 0 and step_n % debug_n == 0:
                    key = f"{atom_1}-{atom_2}-{atom_3}-{atom_4}"
                    if key == "C-N-CA-C":
                        print(f"dih{fa.abs().mean().item():8.3f}", end="  ")
                        print(f"phi{dih_violate.abs().median().item():8.3f}", end="  ")
                    elif key == "CA-C-N-CA":
                        print(f"ome{dih_violate.abs().median().item():8.3f}")

            # Apply integrator
            if integrator == "vel" and step_n < n_steps:
                for atom in atoms:
                    vels[atom] += 0.5 * (accs_last[atom] + accs[atom]) * timestep
                    accs_last[atom] = accs[atom]
            elif integrator == "no_vel" and step_n < n_steps:
                for atom in atoms:
                    coords_next = 2 * self.coords[atom] - coords_last[atom] + accs[atom] * timestep ** 2
                    coords_last[atom] = self.coords[atom]
                    self.coords[atom] = coords_next
            elif integrator == "min" or step_n >= n_steps:
                for atom in atoms:
                    self.coords[atom] += accs[atom].clamp(min=-100.0, max=100.0) * min_speed

            # Apply Andersen thermostat
            # Simulated annealing changes temperature over simulation
            temperature = temperature_start - (temperature_start - temperature_end) * step_n / n_steps
            if thermostat_const > 0.0 and step_n < n_steps:
                thermostat_prob = timestep / thermostat_const
                for ri in range(self.n_res):
                    for atom in atoms:
                        if random() < thermostat_prob:
                            if integrator == "vel":
                                new_vel = torch.randn(self.n_trajs, 3, device=dev) * temperature
                                vels[atom][:, ri] = new_vel
                            elif integrator == "no_vel":
                                new_diff = torch.randn(self.n_trajs, 3, device=dev) * temperature * timestep
                                coords_last[atom][:, ri] = self.coords[atom][:, ri] - new_diff

            if record_n > 0 and step_n % record_n == 0:
                for si in range(self.n_trajs):
                    coords_combined = torch.cat([self.coords[atom][si] for atom in atoms if atom != "H"], dim=0)
                    print("Sim {:2} - Step {:5} - Temp {:6.3f} - Distogram satisfaction {:5.1f}%".format(
                            si + 1, step_n + 1, temperature,
                            100.0 * cb_energies[si].sum() / self.min_cb_energy))

            # Exit if there are any NaN values
            for atom in atoms:
                if torch.isnan(self.coords[atom]).any():
                    print("Encountered a NaN value, exiting")
                    sys.exit()

        # Return mean distogram satisfaction score
        return cb_energies.sum() / (self.min_cb_energy * self.n_trajs)

    def write_coords(self, out_prefix):
        # Write coordinates to a PDB file
        for si in range(self.n_trajs):
            with open(f"{out_prefix}_{si + 1}.pdb", "w") as of:
                ai = 0
                for ri in range(self.n_res):
                    for atom in atoms:
                        if not (atom == "CB" and self.sequence[ri] == "G"):
                            ai += 1
                            of.write("ATOM   {:>4}  {:<2}  {:>3} A{:>4}    {:>8.3f}{:>8.3f}{:>8.3f}  1.00  0.00          {:>2}  \n".format(
                                ai, atom, one_to_three_aas[self.sequence[ri]], ri + 1,
                                self.coords[atom][si, ri, 0].item(), self.coords[atom][si, ri, 1].item(),
                                self.coords[atom][si, ri, 2].item(), elements[atom]))

# Reweight MSA based on cutoff
def reweight(msa1hot, cutoff):
    id_min = msa1hot.shape[1] * cutoff
    id_mtx = torch.einsum("ikl,jkl->ij", msa1hot, msa1hot)
    id_mask = id_mtx > id_min
    w = 1.0 / id_mask.float().sum(dim=-1)
    return w

# Shrunk covariance inversion
def fast_dca(msa1hot, weights, penalty=4.5):
    nr, nc, ns = msa1hot.shape
    x = msa1hot.view(nr, -1)
    num_points = weights.sum() - torch.sqrt(weights.mean())

    mean = (x * weights[:, None]).sum(dim=0, keepdim=True) / num_points
    x = (x - mean) * torch.sqrt(weights[:, None])

    cov = (x.t() @ x) / num_points
    cov_reg = cov + torch.eye(nc * ns, device=dev_nn) * penalty / torch.sqrt(weights.sum())

    inv_cov = torch.inverse(cov_reg)
    x1 = inv_cov.view(nc, ns, nc, ns)
    x2 = x1.transpose(1, 2).contiguous()
    features = x2.reshape(nc, nc, ns * ns)

    x3 = torch.sqrt((x1[:, :-1, :, :-1] ** 2).sum(dim=(1, 3))) * (1 - torch.eye(nc, device=dev_nn))
    apc = x3.sum(dim=0, keepdim=True) * x3.sum(dim=1, keepdim=True) / x3.sum()
    contacts = (x3 - apc) * (1 - torch.eye(nc, device=dev_nn))
    return torch.cat((features, contacts[:, :, None]), dim=2)

# Gaussian weights for distogram forcefield
def gaussian_weights(output):
    # Default value means close residue pairs and those with highest density in the final bin
    #   are ignored when calculating forces
    n_res = output.size(2)
    gaussian_weights = torch.zeros(n_res, n_res, n_bins, device=dev)
    for wi in range(n_res):
        for wj in range(wi + 5, n_res):
            weights = output.data[0, 2:(n_bins + 2), wi, wj].to(dev)
            if torch.argmax(weights) != n_bins - 1:
                gaussian_weights[wi, wj] = -weights
                gaussian_weights[wj, wi] = -weights
    return gaussian_weights

# Predicted hydrogen bond constraints
def hb_constraints(output, hb_prob):
    # Row is hydrogen, column is oxygen
    # Default values mean residue pairs without explicit values have min but not max defined
    n_res = output.size(2)
    hb_dists_min = torch.ones(n_res, n_res, device=dev) * 1.84
    hb_dists_max = torch.ones(n_res, n_res, device=dev) * 1_000_000
    for wi in range(n_res):
        for wj in range(n_res):
            if abs(wi - wj) >= hb_res_sep and output.data[0, 1, wi, wj] >= hb_prob:
                hb_dists_max[wi, wj] = 2.14
    return {"min": hb_dists_min, "max": hb_dists_max}

# Convert predicted dihedrals from bins to constraints
def dihedral_bins_to_constraints(fields, pthresh):
    pmax = 0.0
    for k in range(n_bins):
        if fields[k] > pmax:
            pmax = fields[k]
            kmax = k
    psum = pmax
    k1 = kmax - 1
    k2 = kmax + 1
    lastk1, lastk2 = kmax, kmax
    while psum < pthresh and k2 - k1 < (n_bins - 1):
        p1 = fields[k1 % n_bins]
        p2 = fields[k2 % n_bins]
        if p1 > p2:
            psum += p1
            lastk1 = k1
            k1 -= 1
        else:
            psum += p2
            lastk2 = k2
            k2 += 1
    ang = (kmax + 0.5) * 360.0 / n_bins - 180.0
    sdev = (lastk2 - lastk1 + 1) * 360.0 / (n_bins * 2)
    meets_threshold = psum > pthresh
    return ang, sdev, meets_threshold

# Predicted dihedral constraints
def dihedral_constraints(output, iter_n):
    # Row is hydrogen, column is oxygen
    # Default values mean not all residues have to have constraints
    n_res = output.size(2)
    phis_mean = torch.zeros(n_res, device=dev)
    phis_dev = torch.ones(n_res, device=dev) * 1_000_000
    psis_mean = torch.zeros(n_res, device=dev)
    psis_dev = torch.ones(n_res, device=dev) * 1_000_000

    for wi in range(n_res - 1):
        # Need to clone here as the values are modified
        probs = output.data[0, 36:70, wi, wi + 1].clone()
        ang, sdev, meets_threshold = dihedral_bins_to_constraints(probs,
                        phi_prob_init if iter_n == 1 else phi_prob_iter)
        if meets_threshold:
            phis_mean[wi + 1] = radians(ang)
            phis_dev [wi + 1] = radians(sdev)
        probs = output.data[0, 70:104, wi, wi + 1].clone()
        ang, sdev, meets_threshold = dihedral_bins_to_constraints(probs,
                        psi_prob_init if iter_n == 1 else psi_prob_iter)
        if meets_threshold:
            psis_mean[wi] = radians(ang)
            psis_dev [wi] = radians(sdev)

    # Terminal residues lack certain dihedral angles
    phis_mean = phis_mean[1:]
    phis_dev = phis_dev[1:]
    psis_mean = psis_mean[:-1]
    psis_dev = psis_dev[:-1]
    return {"phis_mean": phis_mean, "phis_dev": phis_dev,
            "psis_mean": psis_mean, "psis_dev": psis_dev}

# Generate full atom models with MODELLER and save the model with the best DOPE score
def modeller_fa_and_score(env, ali_fp, iter_n, output):
    n_res = output.size(2)
    dope_scores = []
    print()
    print("Generating full atom models with MODELLER")
    with open(os.devnull, "w") as f, redirect_stdout(f):
        for ti in range(n_trajs):
            a = automodel(env, alnfile=ali_fp, knowns=[f"traj_{ti + 1}"], sequence="target")
            a.starting_model = 1
            a.ending_model = 1
            a.make()
            for ext in ["D00000001", "V99990001", "ini", "rsr", "sch"]:
                with suppress(FileNotFoundError):
                    os.remove(f"target.{ext}")
            try:
                shutil.move("target.B99990001.pdb", f"traj_{ti + 1}_fa.pdb")
            except FileNotFoundError:
                dope_scores.append(10_000.0)
                continue
            mdl = complete_pdb(env, f"traj_{ti + 1}_fa.pdb")
            atmsel = selection(mdl.chains[0])
            score = atmsel.assess_dopehr()
            dope_scores.append(score)

    print()
    print(f"Iteration {iter_n} MODELLER DOPE scores are:")
    for ti in range(n_trajs):
        print(f"{(ti + 1):<3}  {dope_scores[ti]:10.2f}")
    print()

    if np.min(dope_scores) == 10_000.0:
        print("Could not produce a valid model, exiting")
        sys.exit()

    best_model_n = np.argmin(dope_scores) + 1
    shutil.copyfile(f"traj_{best_model_n}_fa.pdb", f"best_iter_{iter_n}.pdb")
    if not keep_tempdir:
        for ti in range(n_trajs):
            with suppress(FileNotFoundError):
                os.remove(f"traj_{ti + 1}.pdb")
                os.remove(f"traj_{ti + 1}_fa.pdb")
    return np.min(dope_scores)

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
    init_dmap = torch.from_numpy(init_dmap).type(torch.FloatTensor).contiguous().to(dev_nn)
    return init_dmap

def protocol_fdf(aln_file, out_file):
    start_time = datetime.now()
    print("Predicting structure from the alignment in", aln_file)

    network_init = GRUResNetInit(512, 128).eval().to(dev_nn)
    if n_iters > 1:
        network_iter = GRUResNetIter(512, 128).eval().to(dev_nn)

    script_dir = os.path.dirname(os.path.realpath(__file__))

    with open(aln_file, "r") as f:
        aln = f.read().splitlines()

    aa_trans = str.maketrans("ARNDCQEGHILKMFPSTWYVBJOUXZ-.", "ABCDEFGHIJKLMNOPQRSTUUUUUUVV")

    nseqs = len(aln)
    sequence = aln[0]
    length = len(sequence)
    alnmat = (np.frombuffer("".join(aln).translate(aa_trans).encode("latin-1"), dtype=np.uint8) - ord("A")).reshape(nseqs,length)
    print("Sequence has", length, "residues:")
    print(sequence)

    inputs = torch.from_numpy(alnmat).type(torch.LongTensor).to(dev_nn)

    msa1hot = F.one_hot(torch.clamp(inputs, max=20), 21).float()
    w = reweight(msa1hot, cutoff=0.8)

    f2d_dca = fast_dca(msa1hot, w).float() if nseqs > 1 else torch.zeros((length, length, 442), device=dev_nn)
    f2d_dca = f2d_dca.permute(2,0,1).unsqueeze(0)

    inputs2 = f2d_dca

    # Create a temporary directory to work in that will be removed at the end
    for i in count(start=1):
        temp_dir = f"dmpfold_tempdir_{str(i).zfill(4)}"
        try:
            os.mkdir(temp_dir)
            break
        except FileExistsError:
            pass
    print("Will work in temporary directory", temp_dir)
    os.chdir(temp_dir)

    # MODELLER setup
    with open(os.devnull, "w") as f, redirect_stdout(f):
        env = environ()
        env.libs.topology.read(file="$(LIB)/top_heav.lib")
        env.libs.parameters.read(file="$(LIB)/par.lib")

    # Write alignment script used by MODELLER
    ali_fp = "modeller.ali"
    with open(ali_fp, "w") as of:
        of.write(f">P1;target\n")
        of.write(f"sequence:target:1:A:{length}:A:x:x:2.0:-1.0\n")
        of.write(f"{sequence}*\n")
        for i in range(n_trajs):
            of.write("\n")
            of.write(f">P1;traj_{i + 1}\n")
            of.write(f"structureX:traj_{i + 1}:1:A:{length}:A:x:x:2.0:0.02\n")
            of.write(f"{sequence}*\n")

    print()
    print(f"Starting iteration 1 of {n_iters}")
    print()

    network_init.eval()
    with torch.no_grad():
        model_dir_init = os.path.join(script_dir, "nn", "multigram")
        network_init.load_state_dict(torch.load(os.path.join(model_dir_init, "FINAL_fullmap_distcov_model1.pt"),
                                        map_location=dev_nn))
        output = network_init(inputs, inputs2)
        network_init.load_state_dict(torch.load(os.path.join(model_dir_init, "FINAL_fullmap_distcov_model2.pt"),
                                        map_location=dev_nn))
        output += network_init(inputs, inputs2)
        network_init.load_state_dict(torch.load(os.path.join(model_dir_init, "FINAL_fullmap_distcov_model3.pt"),
                                        map_location=dev_nn))
        output += network_init(inputs, inputs2)
        network_init.load_state_dict(torch.load(os.path.join(model_dir_init, "FINAL_fullmap_distcov_model4.pt"),
                                        map_location=dev_nn))
        output += network_init(inputs, inputs2)
        output /= 4
        output[0,0:2,:,:] = F.softmax(output[0,0:2,:,:], dim=0)
        output[0,2:36,:,:] = F.softmax(0.5 * (output[0,2:36,:,:] + output[0,2:36,:,:].transpose(-1,-2)), dim=0)
        output[0,36:70,:,:] = F.softmax(output[0,36:70,:,:], dim=0)
        output[0,70:104,:,:] = F.softmax(output[0,70:104,:,:], dim=0)

    print("Neural network inference done, generating models")
    print("")
    force_folder = ForceFolder(sequence, n_trajs, gaussian_weights(output),
                            hb_constraints(output, hb_prob_init), dihedral_constraints(output, 1))
    satisfaction_score = force_folder.fold(n_steps)
    force_folder.write_coords("traj")
    best_score = modeller_fa_and_score(env, ali_fp, 1, output)
    iter_scores = [best_score]

    for iter_n in range(2, n_iters + 1):
        # Combine standard input with distance map from last iteration best model
        inputs2 = torch.cat((f2d_dca, read_dmap(f"best_iter_{iter_n - 1}.pdb", length)), dim=1)

        print(f"Starting iteration {iter_n} of {n_iters}")
        print()

        network_iter.eval()
        with torch.no_grad():
            model_dir_iter = os.path.join(script_dir, "nn", "multigram_iter")
            network_iter.load_state_dict(torch.load(os.path.join(model_dir_iter, "FINAL_fullmap_distcov_model1.pt"),
                                            map_location=dev_nn))
            output = network_iter(inputs, inputs2)
            network_iter.load_state_dict(torch.load(os.path.join(model_dir_iter, "FINAL_fullmap_distcov_model2.pt"),
                                            map_location=dev_nn))
            output += network_iter(inputs, inputs2)
            network_iter.load_state_dict(torch.load(os.path.join(model_dir_iter, "FINAL_fullmap_distcov_model3.pt"),
                                            map_location=dev_nn))
            output += network_iter(inputs, inputs2)
            output /= 3
            output = torch.max(output, network_iter(inputs, inputs2))
            output[0,0:2,:,:] = F.softmax(output[0,0:2,:,:], dim=0)
            output[0,2:36,:,:] = F.softmax(0.5 * (output[0,2:36,:,:] + output[0,2:36,:,:].transpose(-1,-2)), dim=0)
            output[0,36:70,:,:] = F.softmax(output[0,36:70,:,:], dim=0)
            output[0,70:104,:,:] = F.softmax(output[0,70:104,:,:], dim=0)

        print("Neural network inference done, generating models")
        print("")
        force_folder = ForceFolder(sequence, n_trajs, gaussian_weights(output),
                        hb_constraints(output, hb_prob_iter), dihedral_constraints(output, iter_n))
        satisfaction_score = force_folder.fold(n_steps)
        force_folder.write_coords("traj")
        best_score = modeller_fa_and_score(env, ali_fp, iter_n, output)
        iter_scores.append(best_score)

    if not keep_tempdir:
        os.remove(ali_fp)
    os.chdir("..")
    if final_from_all_iters:
        print("Saving the best scoring model across all iterations to", out_file)
        iter_n_to_use = np.argmin(iter_scores) + 1
    else:
        print("Saving the best scoring model from iteration", n_iters, "to", out_file)
        iter_n_to_use = n_iters
    with open(os.path.join(temp_dir, f"best_iter_{iter_n_to_use}.pdb")) as f:
        with open(out_file, "w") as of:
            of.write(f"REMARK   6 Model generated by DMPfold2 {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
            of.write(f"REMARK   6 Mean distogram satisfaction fraction {satisfaction_score.item():.3f}\n")
            for line in f:
                if line.startswith("ATOM") or line.startswith("TER"):
                    # Relabel blank chain ID to A
                    of.write(line[:21] + "A" + line[22:])
                elif line.startswith("END"):
                    of.write(line)

    if not keep_tempdir:
        print("Deleting temporary directory", temp_dir)
        shutil.rmtree(temp_dir)
    print("Done in", str(datetime.now() - start_time).split(".")[0])
