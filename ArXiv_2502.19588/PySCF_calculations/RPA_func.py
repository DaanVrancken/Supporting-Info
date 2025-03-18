import numpy as np
import gc

from collections.abc import Iterator, Callable
from typing import Union, Optional

from PyFoldHub.helpers.helpers import get_diag_mask
import PyFoldHub.helpers.DF_tools as DF_tools
from PyFoldHub.helpers.helpers import *

from pyscf.ao2mo.incore import _conc_mos
from pyscf.ao2mo import _ao2mo
from pyscf.pbc import df, dft

from opt_einsum import contract, contract_path


au2ev = 27.211386027


def get_weights(
    q_scaled: np.ndarray, shape: tuple, mask: np.ndarray = None, nkpts: int = None
):
    if nkpts is None:
        nkpts = len(q_scaled)
    if mask is None:
        mask = get_diag_mask(shape)
    Q = np.sqrt(
        min(q_scaled[0], 1 - q_scaled[0]) ** 2
        + min(q_scaled[1], 1 - q_scaled[1]) ** 2
        + min(q_scaled[2], 1 - q_scaled[2]) ** 2
    )
    weights = np.ones(shape) * 1 / nkpts
    if Q < 1e-12:
        weights[mask] *= 1#1 / nkpts
    else:
        weights[mask] *= 1 #Q**2 / (Q**2 - 1 / nkpts**2 / 2**2)
    return weights


def get_Lij_GDF(
    kpts: np.ndarray,
    kscaled: np.ndarray,
    cgw,
    mo_coeff_wan: np.ndarray,
    mo_coeff: np.ndarray,
    kL: int = 0,
    nkpts: int = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cgw.logger.debug(
        "Read Lpq (kL: %s / %s)" % (kL + 1, nkpts),
    )
    mydf = cgw._scf.with_df
    nmo = mo_coeff[0].shape[1]
    norbs_wan = mo_coeff_wan[0].shape[1]

    Lij_GDF = []
    Lij_mo_GDF = []

    if nkpts is None:
        kpts = len(kscaled)

    kidx = np.zeros((nkpts), dtype=np.int64)
    kidx_r = np.zeros((nkpts), dtype=np.int64)
    for i, kpti in enumerate(kpts):
        for j, kptj in enumerate(kpts):
            # Find (ki,kj) that satisfies momentum conservation with kL
            cgw.logger.debug(
                "Read Lpq (kL: %s / %s, ki: %s, kj: %s)" % (kL + 1, nkpts, i, j),
            )
            kconserv = -kscaled[i] + kscaled[j] + kscaled[kL]
            is_kconserv = np.linalg.norm(np.round(kconserv) - kconserv) < 1e-12
            if is_kconserv:
                kidx[i] = j
                kidx_r[j] = i
                # logger.debug(
                #     cgw,
                #     "Read Lpq (kL: %s / %s, ki: %s, kj: %s)" % (kL + 1, nkpts, i, j),
                # )
                Lij_out = None
                # Read (L|pq) and ao2mo transform to (L|ij)
                Lpq = []
                Lpq_GDF = []
                Lpq_ft = []
                for LpqR, LpqI, sign in mydf.sr_loop(
                    kpti_kptj=[kpti, kptj],
                    max_memory=0.1 * cgw._scf.max_memory,
                    compact=False,
                ):
                    Lpq_GDF.append(LpqR + LpqI * 1.0j)
                # support uneqaul naux on different k points
                Lpq_GDF = np.vstack(Lpq_GDF).reshape(-1, nmo, nmo)

                Lij_out = None
                moij, ijslice = _conc_mos(mo_coeff[i], mo_coeff[j])[2:]
                tao = []
                ao_loc = None
                Lij_out = _ao2mo.r_e2(
                    Lpq_GDF, moij, ijslice, tao, ao_loc, out=Lij_out
                ).reshape(-1, nmo, nmo)
                Lij_mo_GDF.append(np.array(Lij_out))
                Lij_out = None
                moij, ijslice = _conc_mos(mo_coeff[i], mo_coeff[j])[2:]
                tao = []
                ao_loc = None
                Lij_out = None
                moij, ijslice = _conc_mos(mo_coeff_wan[i], mo_coeff_wan[j])[2:]
                tao = []
                ao_loc = None
                Lij_out = _ao2mo.r_e2(
                    Lpq_GDF, moij, ijslice, tao, ao_loc, out=Lij_out
                ).reshape(-1, norbs_wan, norbs_wan)
                Lij_GDF.append(np.array(Lij_out))

                del Lpq, Lpq_GDF, Lpq_ft, Lij_out
                gc.collect()

    Lij_GDF = np.array(Lij_GDF)
    Lij_mo_GDF = np.array(Lij_mo_GDF)
    return Lij_GDF, Lij_mo_GDF, kidx, kidx_r


def get_Lij_AFT(
    kpts: np.ndarray,
    kscaled: np.ndarray,
    cgw,
    mo_coeff_wan: np.ndarray,
    mo_coeff: np.ndarray,
    kL: int = 0,
    nkpts: int = None,
):
    # kidx: save kj that conserves with kL and ki (-ki+kj+kL=G)
    # kidx_r: save ki that conserves with kL and kj (-ki+kj+kL=G)
    mydf = cgw._scf.with_df
    nmo = mo_coeff[0].shape[1]
    norbs_wan = mo_coeff_wan[0].shape[1]
    q = kpts[kL]

    Lij_AFT = np.zeros((nkpts, cgw.naux_AFT, norbs_wan, norbs_wan), dtype=np.complex128)
    Lkl_AFT = np.zeros((nkpts, cgw.naux_AFT, norbs_wan, norbs_wan), dtype=np.complex128)
    Lij_mo_AFT = np.zeros((nkpts, cgw.naux_AFT, nmo, nmo), dtype=np.complex128)
    Lkl_mo_AFT = np.zeros((nkpts, cgw.naux_AFT, nmo, nmo), dtype=np.complex128)

    kidx = np.zeros((nkpts), dtype=np.int64)
    kidx_r = np.zeros((nkpts), dtype=np.int64)
    for i, kpti in enumerate(kpts):
        for j, kptj in enumerate(kpts):
            # Find (ki,kj) that satisfies momentum conservation with kL
            kconserv = -kscaled[i] + kscaled[j] + kscaled[kL]
            is_kconserv = np.linalg.norm(np.round(kconserv) - kconserv) < 1e-12
            if is_kconserv:
                kidx[i] = j
                kidx_r[j] = i
                # logger.debug(
                #     cgw,
                #     "Read Lpq (kL: %s / %s, ki: %s, kj: %s)" % (kL + 1, nkpts, i, j),
                # )
                L = DF_tools.get_L(
                    mydf,
                    [
                        mo_coeff_wan[i],
                        mo_coeff_wan[j],
                        mo_coeff_wan[j],
                        mo_coeff_wan[i],
                    ],
                    kpts=[kpti, kptj, kptj, kpti],
                    compact=False,
                    q=q,
                    mesh=cgw.mesh,
                )
                Lij_AFT[i] = L[0]
                Lkl_AFT[i] = L[1]
                L = DF_tools.get_L(
                    mydf,
                    [mo_coeff[i], mo_coeff[j], mo_coeff[j], mo_coeff[i]],
                    kpts=[kpti, kptj, kptj, kpti],
                    compact=False,
                    q=q,
                    mesh=cgw.mesh,
                )
                Lij_mo_AFT[i] = L[0]
                Lkl_mo_AFT[i] = L[1]
    return Lij_AFT, Lkl_AFT, Lij_mo_AFT, Lkl_mo_AFT, kidx, kidx_r


def get_U(
    nkpts: int,
    nocc: np.ndarray,
    mo_energy: np.ndarray,
    Lij_mo_GDF: np.ndarray,
    Lij_GDF: np.ndarray,
    exp_U: np.ndarray,
    kidx: np.ndarray,
    kidx_r: np.ndarray,
    wan_tilde: np.ndarray,
    Wmn: np.ndarray,
    Vmn: np.ndarray,
    freqs: np.ndarray = np.array([0]),
    Lij_mo_AFT: np.ndarray = None,
    Lkl_mo_AFT: np.ndarray = None,
    Lij_AFT: np.ndarray = None,
    Lkl_AFT: np.ndarray = None,
    weights: np.ndarray = None,
    memory = "max_input",
    W_GDF_path: list = None,
    W_MDF_path: list = None,
    V_GDF_path: list = None,
    V_MDF_path: list = None,
    df_method: str = "gdf",
    logger = None,
):
    if logger is not None: logger.debug("calculating eps_inv_GDF")
    eps_inv_GDF = get_eps_inv_r_GDF(
        nocc, freqs, mo_energy, Lij_mo_GDF, kidx, wan_tilde, logger=logger, memory = memory
    )[0]
    if df_method == "mdf":
        eps_inv_AFT = get_eps_inv_r_AFT(
            nocc, freqs, mo_energy, Lij_mo_AFT, Lkl_mo_AFT, kidx, wan_tilde, memory = memory
        )[0]
    km = np.arange(nkpts)
    kn = np.arange(nkpts)[kidx]
    ko = np.arange(nkpts)
    kp = np.arange(nkpts)[kidx_r]
    if W_GDF_path is None:
        if logger is not None: logger.debug("Calculating W_GDF_path")
        W_GDF_path = contract_path(
            "KPmn,bK,aK,wPQ,dR,cR,RQop,mnop->wabcdmnop",
            Lij_GDF[km].conj(),
            exp_U[:, kn],
            exp_U[:, km].conj(),
            eps_inv_GDF,
            exp_U[:, ko].conj(),
            exp_U[:, kp],
            Lij_GDF[kp],
            weights,
            optimize=W_GDF_path,
            memory_limit=memory,
        )[0]
        # if logger is not None: logger.debug(W_GDF_path)
    if logger is not None: logger.debug("Calculating Wmnop")
    Wmnop = (
        1
        / nkpts**2
        * au2ev
        * contract(
            "KPmn,bK,aK,wPQ,dR,cR,RQop,mnop->wabcdmnop",
            Lij_GDF[km].conj(),
            exp_U[:, kn],
            exp_U[:, km].conj(),
            eps_inv_GDF,
            exp_U[:, ko].conj(),
            exp_U[:, kp],
            Lij_GDF[kp],
            weights,
            optimize=W_GDF_path,
        )
    )
    if logger is not None: logger.debug("Calculating Wmn")
    Wmn += Wmnop
    if df_method == "mdf":
        if W_MDF_path is None:
            W_MDF_path = contract_path(
                "KPmn, wPQ,bK,aK,dR,cR, RQop, mnop -> wabcdmnop",
                Lkl_AFT[km],
                eps_inv_AFT,
                exp_U[:, km],
                exp_U[:, kn].conj(),
                exp_U[:, ko],
                exp_U[:, kp].conj(),
                Lij_AFT[kp],
                weights,
                optimize="optimal",
            )[0]
        Wmnop = (
            1
            / nkpts**2
            * au2ev
            * contract(
                "KPmn, wPQ,bK,aK,dR,cR, RQop, mnop -> wabcdmnop",
                Lkl_AFT[km],
                eps_inv_AFT,
                exp_U[:, km],
                exp_U[:, kn].conj(),
                exp_U[:, ko],
                exp_U[:, kp].conj(),
                Lij_AFT[kp],
                weights,
                optimize=W_MDF_path,
            )
        )
        Wmn += Wmnop
    if V_GDF_path is None:
        V_GDF_path = contract_path(
            "KPmn,bK,aK,dR,cR,RPop,mnop->abcdmnop",
            Lij_GDF[km].conj(),
            exp_U[:, kn],
            exp_U[:, km].conj(),
            exp_U[:, ko].conj(),
            exp_U[:, kp],
            Lij_GDF[kp],
            weights,
            optimize=V_GDF_path,
        )[0]
    Vmnop = (
        1
        / nkpts**2
        * au2ev
        * contract(
            "KPmn,bK,aK,dR,cR,RPop,mnop->abcdmnop",
            Lij_GDF[km].conj(),
            exp_U[:, kn],
            exp_U[:, km].conj(),
            exp_U[:, ko].conj(),
            exp_U[:, kp],
            Lij_GDF[kp],
            weights,
            optimize=V_GDF_path,
            memory_limit=memory,
        )
    )
    Vmn += Vmnop
    if df_method == "mdf":
        if V_MDF_path is None:
            V_MDF_path = contract_path(
                "KRmn,aK,bK,cS,dS,SRop, mnop -> abcdmnop",
                Lij_AFT[km],
                exp_U[:, km],
                exp_U[:, kn].conj(),
                exp_U[:, ko],
                exp_U[:, kp].conj(),
                Lkl_AFT[kp],
                weights,
                optimize="optimal",
            )[0]
        Vmnop = (
            1
            / nkpts**2
            * au2ev
            * contract(
                "KRmn,aK,bK,cS,dS,SRop, mnop -> abcdmnop",
                Lij_AFT[km],
                exp_U[:, km],
                exp_U[:, kn].conj(),
                exp_U[:, ko],
                exp_U[:, kp].conj(),
                Lkl_AFT[kp],
                weights,
                optimize=V_MDF_path,
            )
        )
        Vmn += Vmnop

    # if crpa.fc:
    #     log.info(f"FC correction: {corr} eV" % corr)

    return Wmn, Vmn, W_GDF_path, W_MDF_path, V_GDF_path, V_MDF_path


def get_U_2(
    nkpts: int,
    nocc: np.ndarray,
    mo_energy: np.ndarray,
    Lij_mo_GDF: np.ndarray,
    Lij_GDF: np.ndarray,
    exp_U: np.ndarray,
    kidx: np.ndarray,
    kidx_r: np.ndarray,
    wan_tilde: np.ndarray,
    Wmn: np.ndarray,
    Vmn: np.ndarray,
    Lij_mo_AFT: np.ndarray = None,
    Lkl_mo_AFT: np.ndarray = None,
    Lij_AFT: np.ndarray = None,
    Lkl_AFT: np.ndarray = None,
    weights: np.ndarray = None,
    memory: int = None,
    df_method: str = "gdf",
):
    eps_inv_GDF = get_eps_inv_r_GDF(nocc, 0, mo_energy, Lij_mo_GDF, kidx, wan_tilde)[0][
        0
    ]
    if df_method == "mdf":
        eps_inv_AFT = get_eps_inv_r_AFT(
            nocc, 0, mo_energy, Lij_mo_AFT, Lkl_mo_AFT, kidx, wan_tilde
        )[0][0]
    for km in range(nkpts):
        # kn = k
        # Find km that conserves with kn and kL (-km+kn+kL=G)
        kn = kidx[km]
        for ko in range(nkpts):
            # ko = kn
            kp = kidx_r[ko]

            Qmn = np.einsum(
                "Pmn,a,b,PQ->abQmn",
                Lij_GDF[km].conj(),
                exp_U[:, kn],
                exp_U[:, km].conj(),
                eps_inv_GDF,
                optimize="optimal",
            )
            Wmnop = (
                1
                / nkpts**2
                * au2ev
                * np.einsum(
                    "abQmn,c,d,Qop,mnop->abcdmnop",
                    Qmn,
                    exp_U[:, kp],
                    exp_U[:, ko].conj(),
                    Lij_GDF[kp],
                    weights,
                    optimize="optimal",
                )
            )
            Wmn += Wmnop
            if df_method == "mdf":
                Qmn = np.einsum(
                    "Pmn,PQ->Qmn", Lkl_AFT[km], eps_inv_AFT, optimize="optimal"
                )
                Wmnop = (
                    1
                    / nkpts**2
                    * au2ev
                    * np.einsum(
                        "Qmn, Qop, mnop -> mnop",
                        Qmn,
                        Lij_AFT[kp],
                        weights,
                        optimize="optimal",
                    )
                )
                Wmn[0, 0, 0, 0] += Wmnop

            Vmnop = (
                1
                / nkpts**2
                * au2ev
                * np.einsum(
                    "Rmn,a,b,c,d,Rop, mnop -> abcdmnop",
                    Lij_GDF[km],
                    exp_U[:, km],
                    exp_U[:, kn].conj(),
                    exp_U[:, ko],
                    exp_U[:, kp].conj(),
                    Lij_GDF[kp].conj(),
                    weights,
                    optimize="optimal",
                )
            )
            Vmn += Vmnop
            if df_method == "mdf":
                Vmnop = (
                    1
                    / nkpts**2
                    * au2ev
                    * np.einsum(
                        "Rmn,Rop, mnop -> mnop",
                        Lij_AFT[km],
                        Lkl_AFT[kp],
                        weights,
                        optimize="optimal",
                    )
                )
                Vmn += Vmnop

    # if crpa.fc:
    #     log.info(f"FC correction: {corr} eV" % corr)

    return Wmn, Vmn


def get_V(
    nkpts: int,
    Lij_GDF: np.ndarray,
    exp_U: np.ndarray,
    kidx: np.ndarray,
    kidx_r: np.ndarray,
    Lij_AFT: np.ndarray = None,
    Lkl_AFT: np.ndarray = None,
    weights: np.ndarray = None,
    memory: int = None,
    df_method: str = "gdf",
):

    km = np.arange(nkpts)
    kn = np.arange(nkpts)[kidx]
    ko = np.arange(nkpts)
    kp = np.arange(nkpts)[kidx_r]
    if V_GDF_path is None:
        V_GDF_path = contract_path(
            "KPmn,bK,aK,dR,cR,RPop,mnop->abcdmnop",
            Lij_GDF[km].conj(),
            exp_U[:, kn],
            exp_U[:, km].conj(),
            exp_U[:, ko].conj(),
            exp_U[:, kp],
            Lij_GDF[kp],
            weights,
            optimize=V_GDF_path,
        )[0]
    Vmnop = (
        1
        / nkpts**2
        * au2ev
        * contract(
            "KPmn,bK,aK,dR,cR,RPop,mnop->abcdmnop",
            Lij_GDF[km].conj(),
            exp_U[:, kn],
            exp_U[:, km].conj(),
            exp_U[:, ko].conj(),
            exp_U[:, kp],
            Lij_GDF[kp],
            weights,
            optimize=V_GDF_path,
            memory_limit=memory,
        )
    )
    Vmn += Vmnop
    if df_method == "mdf":
        if V_MDF_path is None:
            V_MDF_path = contract_path(
                "KRmn,aK,bK,cS,dS,SRop, mnop -> abcdmnop",
                Lij_AFT[km],
                exp_U[:, km],
                exp_U[:, kn].conj(),
                exp_U[:, ko],
                exp_U[:, kp].conj(),
                Lkl_AFT[kp],
                weights,
                optimize="optimal",
            )[0]
        Vmnop = (
            1
            / nkpts**2
            * au2ev
            * contract(
                "KRmn,aK,bK,cS,dS,SRop, mnop -> abcdmnop",
                Lij_AFT[km],
                exp_U[:, km],
                exp_U[:, kn].conj(),
                exp_U[:, ko],
                exp_U[:, kp].conj(),
                Lkl_AFT[kp],
                weights,
                optimize=V_MDF_path,
            )
        )
        Vmn += Vmnop
    return Vmnop


def get_Delta_Sigma_2(
    nkpts: int,
    nocc: np.ndarray,
    Lij_mo_GDF: np.ndarray,
    weights: np.ndarray,
    kidx: np.ndarray,
    freqs: np.ndarray,
    wts: np.ndarray,
    pm: np.ndarray,
    emo: np.ndarray,
    mo_energy: np.ndarray,
    wan_tilde: np.ndarray,
    Delta_Sigma: np.ndarray,
    Lij_mo_AFT: np.ndarray = None,
    Lkl_mo_AFT: np.ndarray = None,
    eps_inv_AFT_r: np.ndarray = None,
    eps_inv_AFT: np.ndarray = None,
    df_method: str = "gdf",
):
    for w_ind, w in enumerate(freqs):
        eps_inv_GDF_r, eps_inv_GDF = get_eps_inv_r_GDF(
            nocc, w, mo_energy, Lij_mo_GDF, kidx, wan_tilde
        )
        eps_inv_GDF_r = eps_inv_GDF_r[0]
        eps_inv_GDF = eps_inv_GDF[0]
        print(f"Calculating for frequency w:{w}")
        # g0 = wts[w] * emo / (emo**2+freqs[w]**2)
        G_L = np.einsum(
            "km, km -> km",
            pm,
            wts[w_ind, None, None] * emo / (emo**2 + w**2),
        )
        G_H = wts[w_ind, None, None] * emo / (emo**2 + w**2) - G_L
        for km in range(nkpts):
            # kn = k
            # Find km that conserves with kn and kL (-km+kn+kL=G)
            kn = kidx[km]

            Qmn_r = np.einsum(
                "Pmn,PQ->Qmn",
                Lij_mo_GDF[km].conj(),
                eps_inv_GDF_r,
                optimize="optimal",
            )
            Wmnno_r = np.einsum(
                "Qmn,Qon, mnno->mno",
                Qmn_r,
                Lij_mo_GDF[km],
                weights,
                optimize="optimal",
            )
            if df_method == "mdf":
                Qmn_r = np.einsum(
                    "Pmn,P->Pmn", Lkl_mo_AFT[km], eps_inv_AFT_r, optimize="optimal"
                )
                Wmnno_r += np.einsum(
                    "Qmn, Qon, mnno -> mno",
                    Qmn_r,
                    Lij_mo_AFT[km],
                    weights,
                    optimize="optimal",
                )

            Qmn = np.einsum(
                "Pmn,PQ->Qmn",
                Lij_mo_GDF[km].conj(),
                eps_inv_GDF,
                optimize="optimal",
            )
            Wmnno = np.einsum(
                "Qmn,Qon,mnno->mno",
                Qmn,
                Lij_mo_GDF[km],
                weights,
                optimize="optimal",
            )
            assert ((np.transpose(Wmnno, (2, 1, 0)) - Wmnno) < 1e-6).all()
            if df_method == "mdf":
                Qmn = np.einsum(
                    "Pmn,P->Pmn", Lkl_mo_AFT[km], eps_inv_AFT, optimize="optimal"
                )
                Wmnno += np.einsum(
                    "Qmn, Qon, mnno -> mno",
                    Qmn,
                    Lij_mo_AFT[km],
                    weights,
                    optimize="optimal",
                )

            Delta_Sigma[0, km, :, :] += (
                -np.einsum("mno,n->mo", Wmnno_r, G_L[kn]) / np.pi / 2
            )
            Delta_Sigma[1, km, :, :] += (
                -np.einsum("mno,n->mo", Wmnno - Wmnno_r, G_L[kn]) / np.pi / 2
            )
            Delta_Sigma[2, km, :, :] += (
                -np.einsum("mno,n->mo", Wmnno, G_H[kn]) / np.pi / 2
            )
    return Delta_Sigma


def get_Delta_Sigma(
    nkpts: int,
    nocc: np.ndarray,
    Lij_mo_GDF: np.ndarray,
    weights: np.ndarray,
    kidx: np.ndarray,
    freqs: np.ndarray,
    wts: np.ndarray,
    pm: np.ndarray,
    mo_energy: np.ndarray,
    ef: float,
    wan_tilde: np.ndarray,
    Delta_Sigma: np.ndarray,
    nw_sigma: int,
    Lij_mo_AFT: np.ndarray = None,
    Lkl_mo_AFT: np.ndarray = None,
    W_GDF_path: list = None,
    W_MDF_path: list = None,
    Delta_path: list = None,
    df_method: str = "gdf",
    memory = "max_input"
):
    w = freqs
    eps_inv_GDF_r, eps_inv_GDF = get_eps_inv_r_GDF(
        nocc, w, mo_energy, Lij_mo_GDF, kidx, wan_tilde, memory=memory
    )
    if df_method == "mdf":
        eps_inv_AFT_r, eps_inv_AFT = get_eps_inv_r_AFT(
            nocc, w, mo_energy, Lij_mo_AFT, Lkl_mo_AFT, kidx, wan_tilde, memory=memory
        )
    print(f"Calculating for frequency w:{w}")
    # g0 = wts[w] * emo / (emo**2+freqs[w]**2)
    G, G_L, G_H, omega = get_Greens_I(
        nkpts,
        nw_sigma,
        mo_energy.shape[1],
        mo_energy,
        ef,
        nocc,
        freqs,
        wts,
        pm,
    )
    km = np.arange(nkpts)
    kn = np.arange(nkpts)[kidx]
    if W_GDF_path is None:
        W_GDF_path = contract_path(
            "KPmn,wPQ,KQon, mnno->Kwmno",
            Lij_mo_GDF[km].conj(),
            eps_inv_GDF_r,
            Lij_mo_GDF[km],
            weights,
            optimize="optimal",
            memory_limit=memory,
        )
        print(W_GDF_path[1])
        W_GDF_path = W_GDF_path[0]
    Wmnno_r = contract(
        "KPmn,wPQ,KQon, mnno->Kwmno",
        Lij_mo_GDF[km].conj(),
        eps_inv_GDF_r,
        Lij_mo_GDF[km],
        weights,
        optimize=W_GDF_path,
    )
    if df_method == "mdf":
        if W_MDF_path is None:
            W_MDF_path = contract_path(
                "KQmn,wQ, KQon, mnno -> Kwmno",
                Lkl_mo_AFT[km],
                eps_inv_AFT_r,
                Lij_mo_AFT[km],
                weights,
                optimize="optimal",
                memory_limit=memory,
            )[0]
        Wmnno_r += contract(
            "KQmn,wQ, KQon, mnno -> Kwmno",
            Lkl_mo_AFT[km],
            eps_inv_AFT_r,
            Lij_mo_AFT[km],
            weights,
            optimize=W_MDF_path,
        )

    Wmnno = contract(
        "KPmn,wPQ,KQon,mnno->Kwmno",
        Lij_mo_GDF[km].conj(),
        eps_inv_GDF,
        Lij_mo_GDF[km],
        weights,
        optimize=W_GDF_path,
    )
    # assert ((np.transpose(Wmnno, (2, 1, 0)) - Wmnno) < 1e-6).all()
    if df_method == "mdf":
        Wmnno += contract(
            "KQmn, wQ, KQon, mnno -> Kwmno",
            Lkl_mo_AFT[km],
            eps_inv_AFT,
            Lij_mo_AFT[km],
            weights,
            optimize=W_MDF_path,
        )
    if Delta_path is None:
        Delta_path = contract_path(
            "Kwmno,Knvw->vmo", Wmnno_r, G_L[kn], optimize="optimal", memory_limit=-1
        )[0]
    Delta_Sigma[0, :, km, :, :] += (
        -contract("Kwmno,Knvw->vmo", Wmnno_r, G_L[kn], optimize=Delta_path) / np.pi
    )
    Delta_Sigma[1, :, km, :, :] += (
        -contract("Kwmno,Knvw->vmo", Wmnno - Wmnno_r, G_L[kn], optimize=Delta_path)
        / np.pi
    )
    Delta_Sigma[2, :, km, :, :] += (
        -contract("Kwmno,Knvw->vmo", Wmnno, G_H[kn], optimize=Delta_path) / np.pi
    )
    return Delta_Sigma, W_GDF_path, W_MDF_path, Delta_path, omega


def get_Greens_I(
    nkpts,
    nw_sigma,
    nmo,
    mo_energy,
    ef,
    nocc,
    freqs,
    wts,
    pm,
):

    # Compute occ for -iw and vir for iw separately
    # to avoid branch cuts in analytic continuation
    omega_occ = np.zeros((nw_sigma), dtype=np.complex128)
    omega_vir = np.zeros((nw_sigma), dtype=np.complex128)
    omega_occ[1:] = -1j * freqs[: (nw_sigma - 1)]
    omega_vir[1:] = 1j * freqs[: (nw_sigma - 1)]

    omega = np.zeros((nkpts, nmo, nw_sigma), dtype=np.complex128)
    for k in range(nkpts):
        omega[k, : nocc[k], 1:] = -1j * freqs[: (nw_sigma - 1)]
        omega[k, nocc[k] :, 1:] = 1j * freqs[: (nw_sigma - 1)]

    emo = omega + ef - mo_energy[:, :, None]

    G = (
        wts[None, None, None, :]
        * emo[:, :, :, None]
        / (emo[:, :, :, None] ** 2 + freqs[None, None, None, :] ** 2)
    )
    G_L = np.einsum(
        "km, kmvw -> kmvw",
        pm,
        G,
    )
    G_H = G - G_L
    return G, G_L, G_H, omega


def get_sigma_R(
    nocc,
    w,
    mo_energy,
    Lij_mo_GDF,
    kidx,
    wan_tilde,
):
    get_eps_inv_r_GDF(nocc, w, mo_energy, Lij_mo_GDF, kidx, wan_tilde, get_GGI)


def get_GG(mo_energy, nocc, i, a, omega, eta=1e-3):
    eia = mo_energy[i, : nocc[i], None] - mo_energy[a, None, nocc[a] :]
    eia = 1.0 / (omega[:, None, None] + eia[None, :, :] + 1j * eta) - 1.0 / (
        omega[:, None, None] - eia[None, :, :] - 1j * eta
    )
    return eia


def get_rho_response(
    nocc: np.ndarray,
    omega,
    mo_energy: np.ndarray,
    Lpq: np.ndarray,
    kidx: np.ndarray,
    Lrs: np.ndarray = None,
    GG_func: Callable = get_GG,
    eta: float = 1e-3,
    memory = "max_input"
):
    nkpts, naux, _, _ = Lpq.shape
    if Lrs is None:
        Lrs = Lpq.conj()
    print("start computing Pi")
    # Compute Pi for kL

    if hasattr(omega, "__len__") == False:
        omega = np.array([omega])
    Pi = np.zeros((len(omega), naux, naux), dtype=np.complex128)
    for i in range(nkpts):
        print("Computing Pi at kpt: ", i)
        # Find ka that conserves with ki and kL (-ki+ka+kL=G)
        a = kidx[i]
        # excitation to the high energy subspace from the hig and low energy subspace below the fermi level.
        # eia = mo_energy[i, : nocc[i], None] - mo_energy[a, None, nocc[a] :]
        # eia = eia[None, :, :] / (omega[:, None, None] ** 2 + (eia * eia)[None, :, :])
        eia = GG_func(mo_energy, nocc, i, a, omega, eta)
        print("eia calculated")
        assert np.isfinite(eia).all()
        Pia = contract(
            "Pia,wia->wPia", Lpq[i][:, : nocc[i], nocc[a] :], eia, memory_limit=memory
        )
        print("Pia calculated")
        # Response from both spin-up and spin-down density
        Pi += (
            4.0
            / nkpts
            * contract(
                "wPia,Qia->wPQ",
                Pia,
                Lrs[i][:, : nocc[i], nocc[a] :],
                memory_limit=memory,
            )
        )
        print("Pi calculated")

    if len(omega) == 1:
        Pi = Pi[0]

    return Pi


def get_GGI(mo_energy, nocc, i, a, omega, eta=None):
    # excitation to the high energy subspace from the hig and low energy subspace below the fermi level.
    eia = mo_energy[i, : nocc[i], None] - mo_energy[a, None, nocc[a] :]
    eia = eia[None, :, :] / (omega[:, None, None] ** 2 + (eia * eia)[None, :, :])
    return eia


def get_rho_response_AFT(
    nocc: np.ndarray,
    omega,
    mo_energy: np.ndarray,
    Lpq: np.ndarray,
    kidx: np.ndarray,
    Lrs: np.ndarray = None,
    GG_func: Callable = get_GG,
    eta: float = 1e-3,
):
    """ """
    nkpts, naux, _, _ = Lpq.shape
    if Lrs is None:
        Lrs = Lpq.conj()
    print("start computing Pi")
    # Compute Pi for kL
    if hasattr(omega, "__len__") == False:
        omega = np.array([omega])
    Pi = np.zeros((len(omega), naux), dtype=np.complex128)
    for i in range(nkpts):
        print("Computing Pi at kpt: ", i)
        # Find ka that conserves with ki and kL (-ki+ka+kL=G)
        a = kidx[i]
        # excitation to the high energy subspace from the hig and low energy subspace below the fermi level.
        # eia = mo_energy[i, : nocc[i], None] - mo_energy[a, None, nocc[a] :]
        # eia = eia[None, :, :] / (omega[:, None, None] ** 2 + (eia * eia)[None, :, :])
        eia = GG_func(mo_energy, nocc, i, a, omega, eta)
        print("eia calculated")
        assert np.isfinite(eia).all()
        Pi += (
            4
            / nkpts
            * np.einsum(
                "Pia,wia,Pia->wP",
                Lpq[i][:, : nocc[i], nocc[a] :],
                eia,
                Lrs[i][:, : nocc[i], nocc[a] :],
                optimize="optimal",
            )
        )
        # print("Pia calculated")
        # Response from both spin-up and spin-down density
        # Pi += 4./nkpts *  np.einsum('Pia,Qia->PQ',Pia,, optimize="optimal")
        print("Pi calculated")
    if len(omega) == 1:
        Pi = Pi[0]

    return Pi


def get_rho_response_sub_weighted(
    nocc: np.ndarray,
    omega,
    mo_energy: np.ndarray,
    Lpq: np.ndarray,
    kidx: np.ndarray,
    wan_tilde: np.ndarray,
    Lrs: np.ndarray = None,
    GG_func: Callable = get_GG,
    eta: float = 1e-3,
    memory = "max_input"
):
    """
    Compute density response function in auxiliary basis at freq iw
    """
    print("using weighted method")
    nkpts, naux, _, _ = Lpq.shape
    if Lrs is None:
        Lrs = Lpq.conj()

    pm = np.sum(np.abs(wan_tilde) ** 2, axis=1)

    # Compute Pi for kL
    if hasattr(omega, "__len__") == False:
        omega = np.array([omega])
    Pi = np.zeros((len(omega), naux, naux), dtype=np.complex128)

    for i in range(nkpts):
        # Find ka that conserves with ki and kL (-ki+ka+kL=G)
        a = kidx[i]
        eia = GG_func(mo_energy, nocc, i, a, omega, eta)
        assert np.isfinite(eia).all()
        Pia = contract(
            "Pia,wia, i, a->wPia",
            Lpq[i][:, : nocc[i], nocc[a] :],
            eia,
            pm[i, : nocc[i]],
            pm[a, nocc[a] :],
            memory_limit = memory,
        )
        # Response from both spin-up and spin-down density
        Pi += (
            4.0
            / nkpts
            * contract(
                "wPia,Qia->wPQ",
                Pia,
                Lrs[i][:, : nocc[i], nocc[a] :],
                memory_limit=memory,
            )
        )
    if len(omega) == 1:
        Pi = Pi[0]
    return Pi


def get_rho_response_sub_weighted_AFT(
    nocc: np.ndarray,
    omega,
    mo_energy: np.ndarray,
    Lpq: np.ndarray,
    kidx: np.ndarray,
    wan_tilde: np.ndarray,
    Lrs: np.ndarray = None,
    GG_func: Callable = get_GG,
    eta: float = 1e-3,
):
    """
    Compute density response function in auxiliary basis at freq iw
    """
    nkpts, naux, _, _ = Lpq.shape
    if Lrs is None:
        Lrs = Lpq.conj()

    pm = np.sum(np.abs(wan_tilde) ** 2, axis=1)

    # Compute Pi for kL
    if hasattr(omega, "__len__") == False:
        omega = np.array([omega])
    Pi = np.zeros((len(omega), naux), dtype=np.complex128)
    for i, kpti in enumerate(kpts):
        # Find ka that conserves with ki and kL (-ki+ka+kL=G)
        a = kidx[i]
        eia = GG_func(mo_energy, nocc, i, a, omega, eta)
        assert np.isfinite(eia).all()
        Pia = np.einsum(
            "Pia,wia, i, a->wPia",
            Lpq[i][:, : nocc[i], nocc[a] :],
            eia,
            pm[i, : nocc[i]],
            pm[a, nocc[a] :],
            optimize="optimal",
        )
        # Response from both spin-up and spin-down density
        Pi += (
            4.0
            / nkpts
            * np.einsum(
                "wPia,Pia->wP",
                Pia,
                Lrs[i][:, : nocc[i], nocc[a] :].conj(),
                optimize="optimal",
            )
        )
    if len(omega) == 1:
        Pi = Pi[0]

    return Pi


def get_rho_response_sub_proj(
    nocc: np.ndarray,
    omega,
    mo_energy: np.ndarray,
    Lpq: np.ndarray,
    kidx: np.ndarray,
    wan_tilde: np.ndarray,
    Lrs: np.ndarray = None,
    GG_func: Callable = get_GG,
    eta: float = 1e-3,
    logger = None, 
    memory = "max_input"
):
    """
    Compute density response function in auxiliary basis at freq iw
    """
    nkpts, naux, nmo, nmo = Lpq.shape
    if logger is not None: logger.debug("Lrs")
    if Lrs is None:
        Lrs = Lpq.conj()
    if logger is not None: logger.debug("Calculating disentanglement projection")
    pmn = np.einsum("klm, kln->kmn", wan_tilde.conj(), wan_tilde)

    # Compute Pi for kL
    if hasattr(omega, "__len__") == False:
        omega = np.array([omega])
    Pi = np.zeros((len(omega), naux, naux), dtype=np.complex128)
    if logger is not None: logger.debug("Calculating Pi_sub r starting kpoint loop")
    
    for i in range(nkpts):
        # Find ka that conserves with ki and kL (-ki+ka+kL=G)
        a = kidx[i]
        A = contract("qm, sn, Pqs -> Pmn", pmn[i, : nocc[i]].conj(), pmn[a, nocc[a] :], Lpq[i][:, : nocc[i], nocc[a] :])
        B = contract("tn, rm, Qrt -> Qmn", pmn[a, nocc[a] :], pmn[i, : nocc[i] ].conj(), Lrs[i][:, : nocc[i], nocc[a] :])
        # if i == 0:
        #     path = contract_path(
        #         "qm, tn, sn, rm, Pqs, Qrt -> PQmn",
        #         pmn[i, : nocc[i]].conj(),
        #         pmn[a, nocc[a] :],
        #         pmn[a, nocc[a] :].conj(),
        #         pmn[i, : nocc[i]],
        #         Lpq[i][:, : nocc[i], nocc[a] :],
        #         Lrs[i][:, : nocc[i], nocc[a] :],
        #         memory_limit=memory,
        #         # optimize="optimal",
        #     )[0]
            
        if logger is not None: logger.debug(f"Calculating eia for kpt: {i}")
        eia = GG_func(mo_energy, nocc, i, a, omega, eta)
        assert np.isfinite(eia).all()
        if logger is not None: logger.debug(f"Calculating proj for kpt: {i}")
        # proj = contract(
        #     "qm, tn, sn, rm, Pqs, Qrt -> PQmn",
        #     pmn[i, : nocc[i]].conj(),
        #     pmn[a, nocc[a] :],
        #     pmn[a, nocc[a] :].conj(),
        #     pmn[i, : nocc[i]],
        #     Lpq[i][:, : nocc[i], nocc[a] :],
        #     Lrs[i][:, : nocc[i], nocc[a] :],
        #     optimize=path,
        # )
        proj = contract("Pmn, Qmn -> PQmn", A, B, memory_limit=memory)
        if i ==0:
            path_pi = contract_path(
            "PQia, wia->wPQ", proj[:, :, : nocc[i], nocc[a] :], eia, memory_limit=memory
            )[0]
        # Response from both spin-up and spin-down density
        if logger is not None: logger.debug(f"Calculating Pi for kpt: {i}")
        Pi += 4.0 / nkpts *contract(
            "PQia, wia->wPQ", proj[:, :, : nocc[i], nocc[a] :], eia, optimize = path_pi
        )
        del proj, eia

    if len(omega) == 1:
        Pi = Pi[0]

    return Pi


def get_rho_response_sub_proj_AFT(
    nocc: np.ndarray,
    omega,
    mo_energy: np.ndarray,
    Lpq: np.ndarray,
    kidx: np.ndarray,
    wan_tilde: np.ndarray,
    Lrs: np.ndarray = None,
    GG_func: Callable = get_GG,
    eta: float = 1e-3,
):
    """
    Compute density response function in auxiliary basis at freq iw
    """
    nkpts, naux, nmo, nmo = Lpq.shape
    if Lrs is None:
        Lrs = Lpq.conj()

    pmn = np.einsum("klm, kln->kmn", wan_tilde.conj(), wan_tilde)

    # Compute Pi for kL
    if hasattr(omega, "__len__") == False:
        omega = np.array([omega])
    Pi = np.zeros((len(omega), naux), dtype=np.complex128)
    for i in range(nkpts):
        # Find ka that conserves with ki and kL (-ki+ka+kL=G)
        a = kidx[i]
        eia = GG_func(mo_energy, nocc, i, a, omega, eta)
        assert np.isfinite(eia).all()
        proj = np.einsum(
            "qm, tn, sn, rm, Pqs, Prt -> Pmn",
            pmn[i, : nocc[i]].conj(),
            pmn[a, nocc[a] :],
            pmn[a, nocc[a] :].conj(),
            pmn[i, : nocc[i]],
            Lpq[i][:, : nocc[i], nocc[a] :],
            Lrs[i][:, : nocc[i], nocc[a] :],
            optimize="optimal",
        )
        Pia = np.einsum(
            "Pia, wia->wP", proj[:, : nocc[i], nocc[a] :], eia, optimize="optimal"
        )
        # Response from both spin-up and spin-down density
        Pi += 4.0 / nkpts * Pia

    if len(omega) == 1:
        Pi = Pi[0]

    return Pi


def get_eps_inv_r_GDF(
    nocc: np.ndarray,
    w,
    mo_energy: np.ndarray,
    Lij_mo_GDF: np.ndarray,
    kidx: np.ndarray,
    wan_tilde: np.ndarray,
    disentangle: str = "proj",
    GG_func: Callable = get_GGI,
    eta: float = 1e-3,
    logger = None, 
    memory = "max_input",
):
    if logger is not None: logger.debug("Calculating Rho_response")
    Pi_GDF = get_rho_response(nocc, w, mo_energy, Lij_mo_GDF, kidx, GG_func=GG_func, memory=memory)
    if logger is not None: logger.debug("Pi matrix succesfully calculated")
    if disentangle == "proj":
        if logger is not None: logger.debug("Calculating Pi_sub_GDF")
        Pi_sub_GDF = get_rho_response_sub_proj(
            nocc, w, mo_energy, Lij_mo_GDF, kidx, wan_tilde, GG_func=GG_func, logger = logger, memory=memory
        )
        if logger is not None: logger.debug("Pi_sub_GDF succesfully calculated")
    elif disentangle == "weighted":
        Pi_sub_GDF = get_rho_response_sub_weighted(
            nocc, w, mo_energy, Lij_mo_GDF, kidx, wan_tilde, GG_func=GG_func, logger= logger, memory=memory
        )
    print("Pi matrix succesfully calculated")
    if logger is not None: logger.debug("Calculating eps_inv_GDF_r")
    eps_inv_GDF_r = np.linalg.inv(
        np.eye(Pi_GDF.shape[1])[None, :, :] - (Pi_GDF - Pi_sub_GDF),
    )
    if logger is not None: logger.debug("eps_inv_GDF_r succesfully calculated")
    if logger is not None: logger.debug("Calculating eps_inv_GDF")
    eps_inv_GDF = np.linalg.inv(np.eye(Pi_GDF.shape[1])[None, :, :] - Pi_GDF)
    if logger is not None: logger.debug("eps_inv_GDF succesfully calculated")
    return eps_inv_GDF_r, eps_inv_GDF


def get_eps_inv_r_AFT(
    nocc: np.ndarray,
    w,
    mo_energy: np.ndarray,
    Lij_mo_AFT: np.ndarray,
    Lkl_mo_AFT: np.ndarray,
    kidx: np.ndarray,
    wan_tilde: np.ndarray,
    disentangle: str = "proj",
):
    Pi_AFT = get_rho_response_AFT(nocc, w, mo_energy, Lij_mo_AFT, kidx, Lkl_mo_AFT)
    if disentangle == "proj":
        Pi_sub_AFT = get_rho_response_sub_proj_AFT(
            nocc,
            w,
            mo_energy,
            Lij_mo_AFT,
            kidx,
            wan_tilde,
            Lkl_mo_AFT,
        )
    elif disentangle == "weighted":
        Pi_sub_AFT = get_rho_response_sub_weighted_AFT(
            nocc,
            w,
            mo_energy,
            Lij_mo_AFT,
            kidx,
            wan_tilde,
            Lkl_mo_AFT,
        )
    eps_inv_AFT_r = 1 / (1 - (Pi_AFT - Pi_sub_AFT))
    eps_inv_AFT = 1 / (1 - Pi_AFT)
    return eps_inv_AFT_r, eps_inv_AFT


def get_hopping(
    kscaled: np.ndarray,
    mo_energy: np.ndarray,
    fermi_energy: float,
    wan_tilde: np.ndarray,
    R_array: np.ndarray,
    nkpts: int = None,
):
    """
    Calculate the hopping matrix elements between Wannier functions.

    Args:
        crpa (object): An instance of the CRPA class.
        wan_coeff (ndarray): The Wannier function coefficients.
        wan_coeff_opt (ndarray): The optimized Wannier function coefficients.
        R_array (ndarray): The array of R vectors.
        orbs_wan (list): The list of Wannier function orbitals.

    Returns:
        ndarray: The hopping matrix elements.

    """
    if nkpts is None:
        nkpts = len(kscaled)
    exp = np.exp(-1j * 2 * np.pi * np.einsum("Rx, kx -> Rk", R_array, kscaled))
    t = (
        -1
        / nkpts
        * np.einsum(
            "Rk, kin, kjn, kn -> Rij",
            exp,
            wan_tilde.conj(),
            wan_tilde,
            (mo_energy - fermi_energy) * au2ev,
        )
    )

    return t


def get_hopping_corr(
    kscaled: np.ndarray,
    occ: np.ndarray,
    wan_tilde: np.ndarray,
    U: np.ndarray,
    zero_ind: int,
    R_array: np.ndarray,
    nkpts: int = None,
):
    """
    Calculate the hopping matrix elements between Wannier functions.

    Args:
        crpa (object): An instance of the CRPA class.
        wan_coeff (ndarray): The Wannier function coefficients.
        wan_coeff_opt (ndarray): The optimized Wannier function coefficients.
        R_array (ndarray): The array of R vectors.
        orbs_wan (list): The list of Wannier function orbitals.

    Returns:
        ndarray: The hopping matrix elements.

    """
    if nkpts is None:
        nkpts = wan_tilde.shape[0]
    exp = np.exp(-1j * 2 * np.pi * np.einsum("Rx, kx -> Rk", R_array, kscaled))
    occ_wan = (
        1
        / nkpts
        * np.einsum(
            "Rk, Pk, kin, kjn ,kn -> PRij",
            exp,
            exp.conj(),
            wan_tilde.conj(),
            wan_tilde,
            occ,
        )
    )
    corr = np.einsum("OPij, ROPmnij -> Rmn", occ_wan, U[0, zero_ind, :, :, :])
    return corr