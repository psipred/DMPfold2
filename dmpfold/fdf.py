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
from PeptideBuilder import PeptideBuilder
from modeller import *
from modeller.automodel import *
from modeller.scripts import complete_pdb

from .networks import aln_to_predictions, aln_to_predictions_iter

dev = "cuda" if torch.cuda.is_available() else "cpu" # Force-directed folding device

n_iters = 3 # Number of iterations
n_trajs = 20 # Number of folding trajectories
n_steps = 20_000 # Steps per folding simulation
gauss_sigma = 1.25 # Width of Gaussians
k_vdw = 100.0 # Force constant for vdw exclusion
n_min_steps = 5_000 # Minimisation steps to finish simulation
min_speed = 0.002 # Minimisation speed
k_dih = 1.0 # Force constant for dihedrals
omega_dev = 0.2 # Omega dihedral deviation
k_hb_dist = 5.0 # Force constant for hydrogen bond distances
hb_res_sep = 10 # Minimum separation for predicted hydrogen bonds

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

# Generate protein models from predicted constraints using force-directed folding
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

    # Run folding simulations for a given number of steps
    def fold(self, n_steps):
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
            forces_min = weight_violate_min * k_hb_dist * violate_min_hb ** pairwise_exponent_min
            forces_max = k_hb_dist * violate_max_hb ** pairwise_exponent_max
            pair_accs = forces_min.unsqueeze(3) * norm_diffs_ho - forces_max.unsqueeze(3) * norm_diffs_ho
            accs["H"] -= pair_accs.sum(dim=2)
            accs["O"] += pair_accs.sum(dim=1)
            if debug_n > 0 and step_n % debug_n == 0:
                print(f"hb{pair_accs.sum(dim=2).abs().mean().item():9.3f}", end="  ")

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

    # Write coordinates to a PDB file
    def write_coords(self, out_prefix):
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
def hbond_constraints(output, hb_prob):
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
def dihedral_bins_to_constraints_fdf(fields, pthresh):
    pmax = 0.0
    for k in range(n_bins):
        if fields[k] > pmax:
            pmax = fields[k]
            kmax = k
    psum = pmax
    k1 = kmax - 1
    k2 = kmax + 1
    lastk1, lastk2 = kmax, kmax
    while psum < pthresh and (k2 - k1) < (n_bins - 1):
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
        ang, sdev, meets_threshold = dihedral_bins_to_constraints_fdf(probs,
                        phi_prob_init if iter_n == 1 else phi_prob_iter)
        if meets_threshold:
            phis_mean[wi + 1] = radians(ang)
            phis_dev [wi + 1] = radians(sdev)
        probs = output.data[0, 70:104, wi, wi + 1].clone()
        ang, sdev, meets_threshold = dihedral_bins_to_constraints_fdf(probs,
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
    for ti in range(n_trajs):
        with suppress(FileNotFoundError):
            os.remove(f"traj_{ti + 1}.pdb")
            os.remove(f"traj_{ti + 1}_fa.pdb")
    return np.min(dope_scores)

# Protein structure prediction with force-directed folding
def aln_to_model_fdf(aln_filepath, out_dir):
    start_time = datetime.now()
    print("Predicting structure from the alignment in", aln_filepath)

    with open(aln_filepath, "r") as f:
        aln = f.read().splitlines()
    sequence = aln[0]
    length = len(sequence)
    print("Sequence has", length, "residues:")
    print(sequence)

    if os.path.isdir(out_dir):
        print(f"Output directory {out_dir} already exists, exiting")
        sys.exit()
    else:
        cwd = os.getcwd()
        os.mkdir(out_dir)
        os.chdir(out_dir)

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

    output = aln_to_predictions(os.path.join(cwd, aln_filepath))

    print("Neural network inference done, generating models")
    print()
    force_folder = ForceFolder(sequence, n_trajs, gaussian_weights(output),
                            hbond_constraints(output, hb_prob_init), dihedral_constraints(output, 1))
    satisfaction_score = force_folder.fold(n_steps)
    force_folder.write_coords("traj")
    best_score = modeller_fa_and_score(env, ali_fp, 1, output)
    iter_scores = [best_score]

    for iter_n in range(2, n_iters + 1):
        print(f"Starting iteration {iter_n} of {n_iters}")
        print()

        output = aln_to_predictions_iter(os.path.join(cwd, aln_filepath), f"best_iter_{iter_n - 1}.pdb")

        print("Neural network inference done, generating models")
        print()
        force_folder = ForceFolder(sequence, n_trajs, gaussian_weights(output),
                        hbond_constraints(output, hb_prob_iter), dihedral_constraints(output, iter_n))
        satisfaction_score = force_folder.fold(n_steps)
        force_folder.write_coords("traj")
        best_score = modeller_fa_and_score(env, ali_fp, iter_n, output)
        iter_scores.append(best_score)

    os.remove(ali_fp)
    if final_from_all_iters:
        iter_n_to_use = np.argmin(iter_scores) + 1
    else:
        iter_n_to_use = n_iters
    with open(f"best_iter_{iter_n_to_use}.pdb") as f, open("final_1.pdb", "w") as of:
        of.write(f"REMARK   6 Model generated by DMPfold2 {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        of.write(f"REMARK   6 Mean distogram satisfaction fraction {satisfaction_score.item():.3f}\n")
        for line in f:
            if line.startswith("ATOM") or line.startswith("TER"):
                # Relabel blank chain ID to A
                of.write(line[:21] + "A" + line[22:])
            elif line.startswith("END"):
                of.write(line)

    os.chdir(cwd)
    print("Writing output to", out_dir)
    print("Done in", str(datetime.now() - start_time).split(".")[0])
