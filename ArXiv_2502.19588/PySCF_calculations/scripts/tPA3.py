from pyscf.pbc import gto
import pyscf
import numpy as np
import os 
import sys
from datetime import datetime
import matplotlib.pyplot as plt

sys.path.append('../../')
from PySCF_calculations.cRPA_run import run_scf, run_bands, run_bands, run_wannier, run_cRPA
from PySCF_calculations.CRPA import KCRPA, get_hopping

cmd_dir = os.getcwd()
cd = os.path.dirname(__file__)

result_map = cd+'/results_{}'.format(sys.argv[0][:-3])
if not os.path.exists(result_map):
    os.mkdir(result_map)
sd = result_map+'/{:%d_%m_%y_%H_%M_%S}'.format(datetime.now())
os.mkdir(sd)
 
cell = gto.Cell()
cell.unit = 'A'

# Set k-points, basis set, disentanglement procedure
kmesh = [8, 1, 1]
cell.basis = 'ccpvtz'
disentangle = "proj"

os.system(f"cp {cd}/properties.py {sd}/.")

a = 2.44924220
cell.a = np.diag([a, 100, 101])
cell.atom = [['C',[0.95468272, 8.66095282, 10.00000000]], 
             ['C',[2.13357544, 9.33904718, 10.00000000]],
             ['H',[0.96243193, 7.57098037, 10.00000000]],
             ['H',[2.12582624, 10.42920196, 10.00000000]]]

cell.spin = 0
cell.precision = 1e-10
cell.build(dimension=1)
kpts = cell.make_kpts(kmesh)
nkpts = kpts.shape[0]
chk_file = "polyacetelyne3_PBE.chk"
cderi_file = False
smearing = False
unrestricted = False
xc = "PBE"
breaksym = False
init_guess = '1e'
auxbasis = None

hspoints = {
"G": [0.0, 0.0, 0.0],
"X": [0.5, 0.0, 0.0],
"M": [0.5, 0.5, 0.0],
"R": [0.5, 0.5, 0.5]}

G = hspoints['G']
X = hspoints['X']
M = hspoints['M']
R = hspoints['R']
kpath = [G, X, G]

xticklabels = ['G', 'X', 'G']

n_band_points = 20
plot_range = [-20, 20]

plot_title = "Band structure polyacetelyne"
plot_wannier = False

num_wann = 2
orbs_wan = np.array([4, 5])
wan_keywords = \
f"""
# PBE energy window
begin projections
C: pz
end projections

guiding_centres = .true.
dist_cutoff_mode = one_dim
one_dim_axis = x

search_shells = 1000

dis_win_min     = -10
dis_win_max     = 5

write_u_matrices = .TRUE.

bands_plot = true

begin kpoint_path
G  0.0 0.0 0.0  X  0.5 0.0 0.0
X  0.5 0.0 0.0  G  0.0 0.0 0.0
end kpoint_path

bands_num_points 15 
"""

PBE = run_scf(sd, cell, kpts, xc, auxbasis, init_guess, breaksym, cderi_file, chk_file, unrestricted)
w90 = run_wannier(sd, cmd_dir, PBE, cell, kmesh, num_wann, wan_keywords, orbs_wan,  unrestricted)
cgw = run_cRPA(sd, PBE, w90, orbs_wan, disentangle)
run_bands(sd, cell, PBE, kpath, plot_wannier, unrestricted, plot_range, plot_title, xticklabels)