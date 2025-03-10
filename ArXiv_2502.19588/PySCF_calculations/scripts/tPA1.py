from PyFoldHub.cRPA_run import run_scf, run_bands, run_bands_2, run_wannier, run_cGW
from PyFoldHub.CRPA import KCRPA, get_hopping

from pyscf.pbc import gto
import pyscf
import numpy as np
import os 
import sys
from datetime import datetime
import matplotlib.pyplot as plt

au2eV = 27.211386245988

cmd_dir = os.getcwd()
cd = os.path.dirname(__file__)

print(sys.argv)

if len(sys.argv) < 2:
    sd = cd+'/results/{:%d_%m_%y_%H_%M_%S}'.format(datetime.now())
    os.mkdir(sd)
else: 
    sd = cmd_dir+'/'+sys.argv[1]
    os.mkdir(sd)
 
cell = gto.Cell()
cell.unit = 'A'


if len(sys.argv) >= 5:
    kmesh = [int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])]
    
else: 
    kmesh = [7, 1, 1]
    
    
if len(sys.argv) >= 6:
    cell.basis = sys.argv[5]
else:
    cell.basis = 'ccpvdz'
    cell.build()
    
if len(sys.argv) >= 7:
    nw = sys.argv[6]
else:
    nw = 1
disentangle = "weighted"

os.system(f"cp {cd}/properties.py {sd}/.")


a = 2.471
cell.a = np.diag([a, a*kmesh[0], a*kmesh[0]])
# cell.a = np.diag([2.455, 1.00, 1.00])
cell.atom = [['C',[0.93699896, 9.99951837, 9.99999960]], 
             ['C',[2.17256226, 10.64948122, 9.99999962]],
             ['H',[0.93681806, 8.90348465, 10.00000025]],
             ['H',[2.17284004, 11.74546053, 10.00000053]]]
# cell.basis = "ccpvdz"
# cell.exp_to_discard=0.1
cell.spin = 0
# cell.ke_cutoff = 1000
# cell.rcut=np.linalg.norm(cell.a[0])*kmesh[0]
cell.precision = 1e-10
cell.build(dimension=1)
kpts = cell.make_kpts(kmesh)
nkpts = kpts.shape[0]
chk_file = "polyacetelyne_PBE.chk"
cderi_file = False
smearing = False
unrestricted = False
xc = "PBE"
breaksym = False
init_guess = '1e'
# auxbasis = 'def2-universal-jfit'
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
# c=0.676, 7.000, 5.500: pz
# c=1.953, 7.686, 5.500: pz
C: pz
end projections

guiding_centres = .true.
dist_cutoff_mode = one_dim
one_dim_axis = x

search_shells = 1000

dis_win_min     = -10
dis_win_max     = 5

# dis_froz_max = 0
# dis_froz_min = -5

write_u_matrices = .TRUE.

# exclude_bands : 1-4, 11-{cell.nao}

bands_plot = true

begin kpoint_path
G  0.0 0.0 0.0  X  0.5 0.0 0.0
X  0.5 0.0 0.0  G  0.0 0.0 0.0
end kpoint_path

bands_num_points 15 
"""

PBE = run_scf(sd, cell, kpts, xc, auxbasis, init_guess, breaksym, cderi_file, chk_file, unrestricted, memory=128000)
w90 = run_wannier(sd, cmd_dir, PBE, cell, kmesh, num_wann, wan_keywords, orbs_wan,  unrestricted)
cgw = run_cGW(sd, PBE, w90, orbs_wan, unrestricted, disentangle, fc =False, nw=nw, memory=128000)
run_bands_2(sd, cell, PBE, kpath, plot_wannier, unrestricted, plot_range, plot_title, xticklabels)
