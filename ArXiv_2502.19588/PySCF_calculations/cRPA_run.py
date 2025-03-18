from PySCF_calculations.CRPA import KCRPA

import numpy as np

from pyscf.pbc import scf, df, dft
from pyscf.lib import chkfile
import PySCF_calculations.pywannier90_s as pyw90
from pyscf.pbc.lib.kpts import KPoints
import time

import matplotlib.pyplot as plt
import copy
import os

# import pyscf.pbc.tools.pyscf_ase as pyscf_ase
from ase.dft.kpoints import sc_special_points as special_points, get_bandpath

au2ev = 27.211386027


def run_scf(
    sd,
    cell,
    kpts,
    xc,
    auxbasis=None,
    init_guess="1e",
    breaksym=None,
    cderi_file=False,
    chk_file=False,
    unrestricted=False,
    soc=False,
    DF="gdf",
    mesh = None, 
    memory = 32000,
):
    kpts_obj = KPoints(cell, kpts).build()
    start = time.time()
    if soc:
        PBE = dft.KGKS(cell, kpts).density_fit(auxbasis=auxbasis).x2c1e()
        PBE.max_cycle = 200
        PBE.max_memory = 32000
        # PBE.breaksym = breaksym
        # PBE.with_x2c.xuncontract = False
        # PBE.with_x2c.approx = 'ATOM1E'
    else:
        if unrestricted:
            PBE = scf.KUKS(cell, kpts)
        else:
            PBE = scf.KRKS(cell, kpts_obj, exxdiv="ewald")
            PBE.breaksym = breaksym

    f = open(sd + "/log.txt", "a")
    PBE.stdout = f
    PBE.xc = xc
    PBE.verbose = 6
    PBE.max_memory = memory
    if DF == "gdf":
        with_df = df.GDF(cell, kpts_obj)
        with_df.exxdiv = "ewald"
    elif DF == "mdf":
        with_df = df.MDF(cell, kpts_obj)
    if auxbasis is not None:
        with_df.auxbasis = auxbasis
    if cderi_file is not False:
        with_df._cderi_to_save = sd + "/" + cderi_file
    if mesh is not None: 
        with_df.mesh = mesh
    with_df = with_df.build(j_only=False)

    if soc:
        PBE.with_x2c.approx = "ATOM1E"

    PBE = PBE.density_fit(with_df=with_df)
    PBE.init_guess = init_guess
    if chk_file is not False:
        PBE.chkfile = sd + "/" + chk_file

    PBE.run()
    end = time.time()
    PBE.dump_flags()
    f.write(f"Time taken for SCF: {end-start} seconds\n")
    return PBE

def load_scf(sd, file, cell, kpts): 
    scf_result_dic = chkfile.load(sd+"/"+file, "scf")
    PBE = scf.KRKS(cell, kpts, exxdiv="ewald")
    PBE.__dict__.update(scf_result_dic)
    with_df = df.GDF(cell, kpts)
    with_df = with_df.build(j_only=False)
    PBE = PBE.density_fit(with_df=with_df)
    f = open(sd + "/log.txt", "w")
    PBE.stdout = f
    return PBE

def run_wannier(
    sd,
    cmd_dir,
    PBE,
    cell,
    kmesh,
    num_wann,
    wan_keywords,
    orbs_wan=None,
    unrestricted=False,
    spin="up",
    supercell=None,
):
    start = time.time()
    print("running wannier...")
    pyw90.save_kmf(PBE, sd + "/chk_W90")
    kmf = pyw90.load_kmf(sd + "/chk_W90")

    # Construct MLWFs:
    w90 = pyw90.W90(kmf, cell, kmesh, num_wann, other_keywords=wan_keywords, spin=spin)

    try:
        w90.kernel()
    except:
        print("memory error")
    if supercell == None:
        supercell = kmesh
    # Export the MWLFs in the .xsf format for plotting with VESTA:
    w90.plot_wf(supercell=supercell, grid=[20, 20, 20], outfile=sd + "/MLFW")

    w90.WANPROJ(filename=sd + "/WANPROJ")

    plt.imshow(np.abs(w90.U_matrix)[0])
    plt.savefig(sd + "/U_matrix.pdf")
    plt.clf()

    os.system(f"mv {cmd_dir}/wan* {sd}/.")

    print("wannier is done")
    end = time.time()
    PBE.stdout.write(f"Time taken for wannier: {end-start} seconds")
    return w90

def run_cRPA(
    sd, PBE, w90, orbs_wan=None, disentangle="proj"
):

    start = time.time()
    crpa = KCRPA(PBE)
    Wmn, Vmn = crpa.kernel(
        wan_coeff=w90.U_matrix,
        wan_coeff_opt=w90.U_matrix_opt,
        disentangle=disentangle,
        orbs_wan=orbs_wan,
    )
    np.save(sd + "/Wmn.npy", Wmn)
    np.save(sd + "/Vmn.npy", Vmn)
    np.save(sd + "/tmn.npy", crpa.t)
    end = time.time()
    PBE.stdout.write(f"Time taken for cRPA: {end-start} seconds")
    return crpa


def run_bands(
    sd,
    cell,
    PBE,
    kpath,
    plot_wannier=False,
    unrestricted=False,
    plot_range=[-10, 10],
    plot_title="Bands",
    xticklabels=None,
):
    start = time.time()
    au2ev = 27.211386245988

    band_kpts = np.loadtxt(sd + "/wannier90_band.kpt", skiprows=1)[:, :3]
    band_kpts = cell.get_abs_kpts(band_kpts)
    wannier_band = np.loadtxt(sd + "/wannier90_band.dat")
    len_wan_band = wannier_band.shape[0]
    ind = int(band_kpts.shape[0])
    labels_co = [
        wannier_band[int(x.split()[1]) - 1, 0]
        for x in open(sd + "/wannier90_band.labelinfo.dat", "r").readlines()
    ]
    labels = [
        x.split()[0]
        for x in open(sd + "/wannier90_band.labelinfo.dat", "r").readlines()
    ]

    E_nk = PBE.get_bands(np.array(band_kpts))
    E_F = PBE.get_fermi()

    plt.figure()
    nbands = cell.nao_nr()

    if unrestricted:
        e_nk_1 = (E_nk[0][0] - E_F[0]) * au2ev
        e_nk_2 = (E_nk[0][1] - E_F[1]) * au2ev
        E_F = E_F[0]
        for n in range(nbands):
            plt.plot(wannier_band[:ind, 0], [e[n] for e in e_nk_1], color="r")
            plt.plot(
                wannier_band[:ind, 0],
                [e[n] for e in e_nk_2],
                color="g",
                linestyle="dotted",
            )
        plt.plot([0, wannier_band[-1, 0]], [0, 0], "--", color="g")
    else:
        e_nk = (E_nk[0] - E_F) * au2ev
        for n in range(nbands):
            plt.plot(wannier_band[:ind, 0], [e[n] for e in e_nk], color="#4169E1")
            plt.plot([0, wannier_band[-1, 0]], [0, 0], "--", color="g")

    print(wannier_band.shape)
    plt.scatter(
        wannier_band[:, 0],
        wannier_band[:, 1] - E_F * au2ev,
        1,
        color="r",
        label="MLFW",
        marker=".",
    )

    plt.xticks(labels_co, labels)
    plt.xlabel("k-vector")
    plt.ylabel(r"$E-E_{F}$ [eV]")
    plt.ylim(plot_range)
    plt.grid()
    plt.title(plot_title)
    plt.savefig(sd + "/Bands.pdf")
    end = time.time()
    PBE.stdout.write(f"Time taken for plotting bands: {end-start} seconds")