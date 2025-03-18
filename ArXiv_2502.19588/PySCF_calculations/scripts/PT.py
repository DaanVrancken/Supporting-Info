from PyFoldHub.cRPA_run import run_scf, run_bands, run_bands, run_wannier, run_cRPA
from PyFoldHub.CRPA import KCRPA, get_hopping

# import sys
# sys.path.append('../../../')
from pyscf import lo
import numpy as np
from pyscf.pbc import gto
import sys
import os 
from datetime import datetime

cmd_dir = os.getcwd()
cd = os.path.dirname(__file__)

result_map = cd+'/results_{}'.format(sys.argv[0][:-3])
if not os.path.exists(result_map):
    os.mkdir(result_map)
sd = result_map+'/{:%d_%m_%y_%H_%M_%S}'.format(datetime.now())
os.mkdir(sd)
 
cell = gto.Cell()   
cell.unit = 'A'
cell.a = np.array([[7.809612770694677, 0, 0], 
                   [0, 23.4995041463670589, 0], 
                   [0, 0, 24.6183938651859293]])
cell.atom = [['C', [5.3428333603131781, 12.9585588091268580, 12.0059921782817334]],
             ['C', [5.8994285303417140, 11.6826454841575895, 12.0083632532141316]],
             ['C', [7.3059312657758895, 11.6819236820107761, 12.0095693036695508]],
             ['C', [0.0545710176350323, 12.9593149255496378, 12.0102387018520318]],
             ['C', [1.4383836804981009, 13.3425606628946589, 12.0093299399273103]],
             ['C', [1.9954674133972761, 14.6182160852969592, 12.0074660985766286]],
             ['C', [3.4018536594935105, 14.6189064559532671, 12.0056863779973177]],
             ['C', [3.9593130219119010, 13.3433496641489864, 12.0056091410428873]],
             ['H', [5.2940602091129811, 10.7767915252709194, 12.0097265649720839]],
             ['H', [0.1010109511087607, 10.7778856864476129, 12.0133003926619342]],
             ['H', [1.3898973307532088, 15.5235740948818002, 12.0069154259971587]],
             ['H', [4.0063392295641780, 15.5251288363021605, 12.0043424370832366]],
             ['S', [6.6039012012530129, 14.1567210320194388, 12.0049821835162405]],
             ['S', [2.6991585058286272, 12.1441114655616680, 12.0079162797617922]]]

# Set k-points, basis set, disentanglement procedure
kmesh = [8, 1, 1]
disentangle = "proj"

kpts = cell.make_kpts(kmesh)
nkpts = kpts.shape[0]
    
cell.basis = {
    'H': gto.basis.parse("""
H  DZVP-GTH-q1 DZVP-GTH
2
1 0 0 4 2
  8.374435000900 -0.028338046100  0.000000000000
  1.805868146000 -0.133381005200  0.000000000000
  0.485252832800 -0.399567606300  0.000000000000
  0.165823693200 -0.553102754100  1.000000000000
2 1 1 1 1
  0.727000000000  1.000000000000
    """),
    
    'C': gto.basis.parse("""
C  DZVP-GTH-q4 DZVP-GTH
2
2 0 1 4 2 2
  4.336237643600  0.149079787200  0.000000000000 -0.087812361900  0.000000000000
  1.288183851300 -0.029264003100  0.000000000000 -0.277556030000  0.000000000000
  0.403776714900 -0.688204051000  0.000000000000 -0.471229509300  0.000000000000
  0.118787765700 -0.396442690600  1.000000000000 -0.405803929100  1.000000000000
3 2 2 1 1
  0.550000000000  1.000000000000
    """),
    
    'S': gto.basis.parse("""
S  DZVP-GTH-q6 DZVP-GTH
2
3 0 1 4 2 2
  1.837962957800  0.383214289100  0.000000000000  0.122135829600  0.000000000000
  1.035773008400 -0.168225731500  0.000000000000 -0.275200246100  0.000000000000
  0.329796987500 -0.825848816600  0.000000000000 -0.572905459200  0.000000000000
  0.107353547100 -0.283275805200  1.000000000000 -0.382546813700  1.000000000000
3 2 2 1 1
  0.479000000000  1.000000000000
    """)}

cell.pseudo = {
    'H': gto.pseudo.parse("""
    H  GTH-PBE-q1 GTH-GGA-q1
1 0 0 0
      0.20059317   2    -4.17806832     0.72440924
     0
    """),
    'C': gto.pseudo.parse("""
C  GTH-PBE-q4 GTH-GGA-q4
2 2 0 0
      0.33855480   2    -8.80455195     1.33837678
     1
      0.30260968   1     9.62286250
    """),
    'S': gto.pseudo.parse("""
S  GTH-PBE-q6 GTH-GGA-q6
2 4 0 0
      0.42011499   1    -5.97478039
     2
      0.36482481   2    13.14466045    -4.24182995
                                        5.47617797
      0.40947909   1     3.70086987

    """)}

os.system(f"cp {cd}/properties.py {sd}/.")

cell.exp_to_discard=0.1
cell.spin = 0
cell.build(dimension=1)

chk_file = "polythiophene_PBE.chk"
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

n_band_points = 100
plot_range = [-20, 20]

plot_title = "Band structure polythiophene"
plot_wannier = False

num_wann = 6
orbs_wan = np.array([20, 21, 22, 23, 24, 25, 26, 27])
wan_keywords = \
f""" 
# PBE energy window
exclude_bands : 1-20, 29-{cell.nao}

num_elec_per_state=2

begin projections
c=1.4383836804981009, 13.3425606628946589, 12.0093299399273103: pz
c=3.9593130219119010, 13.3433496641489864, 12.0056091410428873: pz
c=0.0545710176350323, 12.9593149255496378, 12.0102387018520318: pz
c=5.3428333603131781, 12.9585588091268580, 12.0059921782817334: pz
S: pz
end projections

guiding_centres = .true.
write_u_matrices = .TRUE.

bands_plot = true

begin kpoint_path
G  0.0 0.0 0.0  X  0.5 0.0 0.0
X  0.5 0.0 0.0  G  0.0 0.0 0.0
end kpoint_path

bands_num_points 50 
"""

PBE = run_scf(sd, cell, kpts, xc, auxbasis, init_guess, breaksym, cderi_file, chk_file, unrestricted, memory=120000)
w90 = run_wannier(sd, cmd_dir, PBE, cell, kmesh, num_wann, wan_keywords, orbs_wan,  unrestricted)
crpa = run_cRPA(sd, PBE, w90, orbs_wan)
run_bands(sd, cell, PBE, kpath, plot_wannier, unrestricted, plot_range, plot_title, xticklabels)