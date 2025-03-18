from pyscf import lib
import pyscf
import numpy as np
import gc

from pyscf.lib import logger
from pyscf.ao2mo import _ao2mo
from pyscf.pbc import df, dft
from pyscf.ao2mo.incore import _conc_mos
from pyscf.pbc.tools import pbc as pbctools
import PySCF_calculations.helpers.DF_tools as DF_tools
from PySCF_calculations.helpers.helpers import num_elements_mb
from pyscf.pbc.lib.kpts import KPoints
from PySCF_calculations.RPA_func import *
import copy

gc.set_threshold(0)
einsum = lib.einsum


def kernel(
    crpa,
    naux=None,
    wan_tilde=None,
    disentangle="proj",
    orbs_wan=None,
    kptlist=None,
    nw=None,
    verbose=logger.NOTE,
):
    log = logger.Logger(crpa.stdout, crpa.verbose)

    if crpa.frozen is None:
        frozen = 0
    else:
        frozen = crpa.frozen
    assert frozen == 0

    if orbs_wan is None:
        orbs_wan = np.array([])
    if kptlist is None:
        kptlist = range(crpa.nkpts)

    norbs_wan = wan_tilde.shape[1]
    nkpts = crpa.nkpts

    if getattr(crpa.with_df, "_cderi", None) is None:
        raise RuntimeError(
            "Found incompatible integral scheme %s."
            "KGWAC can be only used with GDF integrals" % crpa.with_df.__class__
        )

    mo_energy = np.array(crpa._scf.mo_energy)
    mo_coeff = np.array(crpa._scf.mo_coeff)
    nmo = crpa.nmo
    nocc = crpa.nocc
    kpts = crpa.kpts
    n_R = len(crpa.sites_U)

    mask = get_diag_mask((norbs_wan, norbs_wan, norbs_wan, norbs_wan))
    mask_mo = get_diag_mask((nmo, nmo, nmo, nmo))

    mo_coeff_wan = contract("kij, knj -> kin", mo_coeff, wan_tilde)

    # possible kpts shift center
    kscaled = crpa.mol.get_scaled_kpts(crpa._scf.kpts)
    kscaled -= kscaled[0]

    exp_U = np.exp(1j * 2 * np.pi * contract("Rx, kx -> Rk", crpa.sites_U, kscaled))
    exp_t = np.exp(1j * 2 * np.pi * contract("Rx, kx -> Rk", crpa.sites, kscaled))

    W_GDF_path = None
    W_MDF_path = None
    W_GDF_path_1 = None
    W_MDF_path_1 = None
    V_GDF_path = None
    V_MDF_path = None
    Delta_path = None

    Wmn = np.zeros(
        (1, n_R, n_R, n_R, n_R, norbs_wan, norbs_wan, norbs_wan, norbs_wan),
        dtype=np.complex256,
    )
    Vmn = np.zeros(
        (n_R, n_R, n_R, n_R, norbs_wan, norbs_wan, norbs_wan, norbs_wan),
        dtype=np.complex256,
    )

    corr = {}
    for kL in range(nkpts):
        crpa.logger.debug(f"Starting kL = {kL}")
        q = kpts[kL]
        q_scaled = kscaled[kL]
        weights = get_weights(
            q_scaled,
            (norbs_wan, norbs_wan, norbs_wan, norbs_wan),
            mask=mask,
            nkpts=nkpts,
        )
        weights_mo = get_weights(
            q_scaled, (nmo, nmo, nmo, nmo), mask=mask_mo, nkpts=nkpts
        )


        Lij_GDF, Lij_mo_GDF, kidx, kidx_r = get_Lij_GDF(
            kpts, kscaled, crpa, mo_coeff_wan, mo_coeff, kL=kL, nkpts=nkpts
        )

        print("Lpq matrix succesfully calculated")
        crpa.logger.debug("Lpq matrix succesfully calculated")
        crpa.logger.debug("Starting Wmn and Vmn calculation")
        Wmn, Vmn, W_GDF_path_1, W_MDF_path_1, V_GDF_path, V_MDF_path = get_U(
            nkpts,
            nocc,
            mo_energy,
            Lij_mo_GDF,
            Lij_GDF,
            exp_U,
            kidx,
            kidx_r,
            wan_tilde,
            Wmn,
            Vmn,
            freqs=np.array([0]),
            weights=weights,
            W_GDF_path=W_GDF_path_1,
            W_MDF_path=W_MDF_path_1,
            V_GDF_path=V_GDF_path,
            V_MDF_path=V_MDF_path,
            logger= crpa.logger,
            memory=crpa.max_elements
        )
        crpa.logger.debug("Wmn and Vmn calculation done")
        

        del (
            Lij_GDF,
            Lij_mo_GDF,
        )
        gc.collect()

    corr["hartree"] = get_hopping_corr(
        crpa.kscaled,
        np.array(crpa._scf.get_occ()),
        wan_tilde,
        Wmn,
        crpa.ind_0_U,
        crpa.sites_U,
    )
    return Wmn, Vmn, corr


class KCRPA(lib.StreamObject):
    _keys = {
        "linearized",
        "ac",
        "fc",
        "frozen",
        "mol",
        "with_df",
        "kpts",
        "nkpts",
        "mo_energy",
        "mo_coeff",
        "mo_occ",
        "sigma",
    }

    def __init__(self, mf, frozen=0):
        self.mol = mf.mol
        self._scf = mf
        self.verbose = self._scf.verbose
        self.stdout = self._scf.stdout
        self.max_mem = mf.max_memory
        self.logger = logger.Logger(self.stdout, self.verbose)

        # TODO: implement frozen orbs
        if frozen > 0:
            raise NotImplementedError
        self.frozen = frozen

        # DF-KGW must use GDF integrals
        if getattr(mf, "with_df", None):
            self.with_df = mf.with_df
        else:
            raise NotImplementedError

        ##################################################
        # don't modify the following attributes, they are not input options
        self._nocc = None
        self._nmo = None
        if isinstance(mf.kpts, KPoints):
            self.kpts = mf.kpts.kpts
        else:
            self.kpts = mf.kpts
        self.kscaled = self.mol.get_scaled_kpts(self.kpts)
        self.mo_energy = np.array(mf.mo_energy)
        self.fermi_energy = mf.get_fermi()
        self.nkpts = len(self.kpts)
        print("nkpts: ", self.nkpts)
        # self.mo_energy: GW quasiparticle energy, not scf mo_energy
        self.mo_energy = None
        self.mo_coeff = mf.mo_coeff
        self.mo_occ = mf.mo_occ
        self.mo_occ_kpts = np.array(mf.get_occ())
        print("shape mo_occ", np.array(self.mo_occ).shape)
        self.sigma = None

        self.t = None
        self.Wmn = None
        self.Vmn = None

        self.wan_coeff = None
        self.orbs_wan = None
        self.overlap = None
        self.nw = None
        self.Nw = None
        self.aux = None
        self.disentangle = "proj"
        self.fc = False
        self.mesh = None
        self._ke_cutoff = 0
        self.sites = np.array(
            [
                [-3, 0, 0],
                [-2, 0, 0],
                [-1, 0, 0],
                [0, 0, 0],
                [1, 0, 0],
                [2, 0, 0],
                [3, 0, 0],
            ]
        )
        self.n_R = len(self.sites)
        self.sites_U = copy.deepcopy(self.sites)
        self.n_RU = len(self.sites_U)
        self.Nw_occ = 1
        self.ind_0_U = np.where(
            (self.sites_U[:, 0] == 0)
            & (self.sites_U[:, 1] == 0)
            & (self.sites_U[:, 2] == 0)
        )[0][0]
        self.ind_0_t = np.where(
            (self.sites[:, 0] == 0) & (self.sites[:, 1] == 0) & (self.sites[:, 2] == 0)
        )[0][0]
        if isinstance(self.with_df, pyscf.pbc.df.df.GDF):
            self.df_method = "gdf"
        if isinstance(self.with_df, pyscf.pbc.df.mdf.MDF):
            self.df_method = "mdf"
            self.ke_cutoff = df.aft.estimate_ke_cutoff(
                self._scf.cell, self._scf.cell.precision
            )

    def dump_flags(self):
        log = logger.Logger(self.stdout, self.verbose)
        log.info("")
        log.info("******** %s ********", self.__class__)
        log.info("method = %s", self.__class__.__name__)
        if self.frozen is not None:
            log.info("frozen orbitals %s", str(self.frozen))
        log.info("number of k-points: %d", self.nkpts)
        log.info(f"number of occupied orbitals: {self.nocc}")
        log.info("number of mo: %d", self.nmo)
        log.info("aux basis: %s", self.with_df.auxbasis)
        log.info("naux: %d", self.naux)
        log.info("ke_cutoff: %f", self.ke_cutoff)
        log.info(f"mesh: {self.mesh}")
        return self

    def dump_summary(self, n=6):
        log = logger.Logger(self.stdout, self.verbose)
        log.info("")
        log.info("******** %s ********", self.__class__)
        log.info("method = %s", self.__class__.__name__)
        log.info(f"number of k-points: {self.nkpts}")
        log.info(f"SCF basis: {self._scf.cell.basis}")
        log.info(f"number of scf basis functions: {self.nmo}")
        log.info(f"aux basis: {self.with_df.auxbasis}")
        log.info(f"number of df aux basis functions: {self.naux}")
        log.info(f"wannier orbitals: {self.orbs_wan}")
        log.info(f"disentanglement method: {self.disentangle}")
        log.info(f"number of occupied orbitals: {self.nocc}")
        log.info(f"nw: {self.nw}")
        if (self.Wmn is not None) and (self.Vmn is not None):
            log.info("Hubbard U-parameters (W_wan) and Coulomb repulsion (V_wan)")
            for i in range(self.Nw):
                for j in range(self.Nw):
                    for k in range(self.Nw):
                        for l in range(self.Nw):
                            log.info(
                                f"W_wan{i+1}{j+1}{k+1}{l+1}={np.round(self.Wmn[0, 0, 0, 0, 0, i, j, k, l], n):<18} V_wan{i+1}{j+1}{k+1}{l+1}={np.round(self.Vmn[0, 0, 0, 0, i, j, k, l], n):<18}"
                            )
        else:
            log.info("kernel not yet computed")
        log.info("hopping terms")
        if self.sites is not None:
            for i_ind, i in enumerate(self.sites):
                log.info(f"hopping terms for site {i}")
                for j in range(self.Nw):
                    for k in range(self.Nw):
                        log.info(f"t{j+1}{k+1}{i}={self.t[i_ind, j, k]}")
        else:
            log.info("hopping terms not yet computed")
        return self

    
    @property
    def nocc(self):
        return np.sum(self.mo_occ_kpts // 2, axis=-1).astype(int)

    @nocc.setter
    def nocc(self, n):
        self._nocc = n

    @property
    def nmo(self):
        return self.mo_occ_kpts.shape[-1]

    @nmo.setter
    def nmo(self, n):
        self._nmo = n

    @property
    def ke_cutoff(self):
        return self._ke_cutoff

    @ke_cutoff.setter
    def ke_cutoff(self, ke_cutoff):
        self._ke_cutoff = ke_cutoff
        mesh = pbctools.cutoff_to_mesh(self._scf.cell.a, ke_cutoff)
        if self._scf.cell.dimension < 3:
            mesh[self._scf.cell.dimension :] += np.mod(
                mesh[self._scf.cell.dimension :], 2
            )
        self.mesh = mesh

    @property
    def max_elements(self): 
        return num_elements_mb(np.complex128, self.max_mem)
        

    run_kernel = kernel

    def kernel(
        self,
        wan_coeff=None,
        wan_coeff_opt=None,
        orbs_wan=None,
        disentangle="proj",
        nw=None,
        verbose=logger.NOTE,
    ):
        """
        Input:
            kptlist: self-energy k-points
            orbs: self-energy orbs
            nw: grid number
        Output:
            mo_energy: GW quasiparticle energy
        """
        self.wan_coeff = wan_coeff

        self.orbs_wan = orbs_wan
        if nw is not None:
            self.nw = int(nw)
        self.Nw = wan_coeff.shape[-1]
        self.disentangle = disentangle

        if wan_coeff_opt.shape != (self.nkpts, self.Nw, self.nmo):
            assert (wan_coeff_opt.shape[0], wan_coeff_opt.shape[1]) == (
                self.nkpts,
                self.Nw,
            )
            self.wan_coeff_opt = np.zeros(
                (self.nkpts, self.Nw, self.nmo), dtype=np.complex128
            )
            self.wan_coeff_opt[:, :, orbs_wan] += wan_coeff_opt
        else:
            self.wan_coeff_opt = wan_coeff_opt
        self.wan_tilde = np.einsum(
            "kni, kij -> knj", self.wan_coeff, self.wan_coeff_opt
        )

        print("Starting kernel...")
        mo_coeff = np.array(self._scf.mo_coeff)
        mo_energy = np.array(self._scf.mo_energy)
        print("mo_energy: ", mo_energy.shape)
        print("mo_coeff: ", mo_coeff.shape)
        print("getting auxilliary basis set")
        if self.mesh is None:
            self.mesh = self.with_df.mesh
        self.naux = df.DF.get_naoaux(self.with_df)
        if self.df_method == "mdf":
            self.naux += np.prod(np.asarray(self.mesh))
        self.naux_GDF = df.DF.get_naoaux(self.with_df)
        if self.df_method == "mdf":
            self.naux_AFT = np.prod(np.asarray(self.mesh))

        self.dump_flags()
        print("Calculating U-parameters")
        Wmn, Vmn, corr = self.run_kernel(
            wan_tilde=self.wan_tilde,
            naux=self.naux,
            orbs_wan=orbs_wan,
            disentangle=disentangle,
            nw=self.nw,
            verbose=self.verbose,
        )
        self.corr = corr
        self.t = get_hopping(
            self.kscaled, mo_energy, self.fermi_energy, self.wan_tilde, self.sites
        )
        self.Vmn = Vmn
        self.Wmn = Wmn
        self.dump_summary()
        return Wmn, Vmn