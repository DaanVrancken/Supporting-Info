import numpy as np
from pyscf import lib
from pyscf.ao2mo import _ao2mo
from pyscf.ao2mo.incore import iden_coeffs, _conc_mos
from pyscf.pbc.df.fft_ao2mo import _format_kpts, _iskconserv
from pyscf.pbc.df.df_ao2mo import _mo_as_complex, _dtrans, _ztrans
from pyscf.pbc.df.df_ao2mo import warn_pbc2d_eri
from pyscf.pbc.lib.kpts_helper import is_zero, gamma_point
from pyscf import __config__


def get_L(
    mydf,
    mo_coeffs,
    kpts=None,
    compact=getattr(__config__, "pbc_df_ao2mo_general_compact", True),
    q=None,
    mesh=None,
):
    warn_pbc2d_eri(mydf)

    cell = mydf.cell
    kptijkl = _format_kpts(kpts)
    kpti, kptj, kptk, kptl = kptijkl
    if isinstance(mo_coeffs, np.ndarray) and mo_coeffs.ndim == 2:
        mo_coeffs = (mo_coeffs,) * 4
    if not _iskconserv(cell, kptijkl):
        lib.logger.warn(
            cell,
            "aft_ao2mo: momentum conservation not found in " "the given k-points %s",
            kptijkl,
        )
        return np.zeros([mo.shape[1] for mo in mo_coeffs])
    if q is None:
        q = kptj - kpti
    if mesh is None:
        mesh = mydf.mesh
    coulG = mydf.weighted_coulG(q, False, mesh)
    all_real = not any(np.iscomplexobj(mo) for mo in mo_coeffs)
    max_memory = max(2000, (mydf.max_memory - lib.current_memory()[0]) * 0.5)
    print(f"q={q}")
    ####################
    # gamma point, the integral is real and with s4 symmetry
    if gamma_point(kptijkl) and all_real:
        # if False:
        print("Gamma point")
        ijmosym, nij_pair, moij, ijslice = _conc_mos(
            mo_coeffs[0], mo_coeffs[1], compact
        )
        klmosym, nkl_pair, mokl, klslice = _conc_mos(
            mo_coeffs[0], mo_coeffs[1], compact
        )
        eri_mo = np.zeros((nij_pair, nkl_pair))
        sym = iden_coeffs(mo_coeffs[0], mo_coeffs[0]) and iden_coeffs(
            mo_coeffs[1], mo_coeffs[1]
        )

        ijR = ijI = klR = klI = buf = None
        Lij = []
        Lkl = []
        for pqkR, pqkI, p0, p1 in mydf.pw_loop(
            mesh, kptijkl[:2], q, max_memory=max_memory, aosym="s2"
        ):
            buf = lib.transpose(pqkR, out=buf)
            ijR, klR = _dtrans(
                buf, ijR, ijmosym, moij, ijslice, buf, klR, klmosym, mokl, klslice, sym
            )
            lib.ddot(ijR.T, klR * coulG[p0:p1, None], 1, eri_mo, 1)
            buf = lib.transpose(pqkI, out=buf)
            ijI, klI = _dtrans(
                buf, ijI, ijmosym, moij, ijslice, buf, klI, klmosym, mokl, klslice, sym
            )
            lib.ddot(ijI.T, klI * coulG[p0:p1, None], 1, eri_mo, 1)
            Lij.append(ijR + ijI * 1j)
            Lkl.append(klR + klI * 1j) * coulG[p0:p1, None]
            pqkR = pqkI = None

    ####################
    # (kpt) i == j == k == l != 0
    # (kpt) i == l && j == k && i != j && j != k  =>
    #
    # elif is_zero(kpti-kptl) and is_zero(kptj-kptk):
    elif False:
        print("i == j == k == l")
        mo_coeffs = _mo_as_complex(mo_coeffs)
        nij_pair, moij, ijslice = _conc_mos(mo_coeffs[0], mo_coeffs[1])[1:]
        nlk_pair, molk, lkslice = _conc_mos(mo_coeffs[0], mo_coeffs[1])[1:]
        eri_mo = np.zeros((nij_pair, nlk_pair), dtype=np.complex128)
        sym = iden_coeffs(mo_coeffs[0], mo_coeffs[0]) and iden_coeffs(
            mo_coeffs[1], mo_coeffs[1]
        )

        zij = zlk = buf = None
        Lij = []
        Lkl = []
        for pqkR, pqkI, p0, p1 in mydf.pw_loop(
            mesh, kptijkl[:2], q, max_memory=max_memory
        ):
            buf = lib.transpose(pqkR + pqkI * 1j, out=buf)
            zij, zlk = _ztrans(buf, zij, moij, ijslice, buf, zlk, molk, lkslice, sym)
            lib.dot(zij.T, zlk.conj() * coulG[p0:p1, None], 1, eri_mo, 1)
            pqkR = pqkI = None
            Lij.append(zij)
            Lkl.append(zlk * coulG[p0:p1, None])

    ####################
    # aosym = s1, complex integrals
    #
    # If kpti == kptj, (kptl-kptk)*a has to be multiples of 2pi because of the wave
    # vector symmetry.  k is a fraction of reciprocal basis, 0 < k/b < 1, by definition.
    # So  kptl/b - kptk/b  must be -1 < k/b < 1.  =>  kptl == kptk
    #
    else:
        print("Complex integrals")
        # mo_coeffs = _mo_as_complex(mo_coeffs)
        nij_pair, moij, ijslice = _conc_mos(mo_coeffs[0], mo_coeffs[1])[1:]
        nkl_pair, mokl, klslice = _conc_mos(mo_coeffs[2], mo_coeffs[3])[1:]
        eri_mo = np.zeros((nij_pair, nkl_pair), dtype=np.complex128)

        tao = []
        ao_loc = None
        zij = zkl = buf = None
        Lij = []
        Lkl = []
        for (pqkR, pqkI, p0, p1), (rskR, rskI, q0, q1) in lib.izip(
            mydf.pw_loop(mesh, kptijkl[:2], q, max_memory=max_memory * 0.1),
            mydf.pw_loop(mesh, -kptijkl[2:], q, max_memory=max_memory * 0.1),
        ):
            buf = lib.transpose(pqkR + pqkI * 1j, out=buf)
            zij = _ao2mo.r_e2(buf, moij, ijslice, tao, ao_loc, out=zij)
            buf = lib.transpose(rskR - rskI * 1j, out=buf)
            zkl = _ao2mo.r_e2(buf, mokl, klslice, tao, ao_loc, out=zkl)

            zij *= coulG[p0:p1, None]
            lib.dot(zij.T, zkl, 1, eri_mo, 1)
            pqkR = pqkI = rskR = rskI = None
            Lij.append(zij)
            Lkl.append(zkl)
        del buf
    Lij = np.vstack(Lij).reshape(-1, mo_coeffs[0].shape[-1], mo_coeffs[1].shape[-1])
    Lkl = np.vstack(Lkl).reshape(-1, mo_coeffs[2].shape[-1], mo_coeffs[3].shape[-1])
    return Lij, Lkl


def transform_L(mo_coeff_1, mo_coeff_2, Lpq):
    nmo_1 = mo_coeff_1.shape[1]
    nmo_2 = mo_coeff_2.shape[1]
    Lij_out = None
    moij, ijslice = _conc_mos(mo_coeff_1, mo_coeff_2)[2:]
    tao = []
    ao_loc = None
    Lij_out = _ao2mo.r_e2(Lpq, moij, ijslice, tao, ao_loc, out=Lij_out).reshape(
        -1, nmo_1, nmo_2
    )
    return np.array(Lij_out)