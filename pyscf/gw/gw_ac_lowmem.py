#!/usr/bin/env python
# Copyright 2014-2026 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

'''
Low-memory variant of spin-restricted G0W0 with analytic continuation (pyscf.gw.gw_ac).

Same theory, grids, analytic continuation and results as pyscf.gw.gw_ac.GWAC (to rounding);
only the evaluation of the correlation self-energy on the imaginary axis is reorganized so
that its memory is bounded by max_memory:

  * The response function Pi(i w) is accumulated over blocks of occupied orbitals (rows of
    the occupied-virtual DF block), so no copy of L_ia and no L_ia-sized temporary is made.
  * W_mn and the self-energy are accumulated over blocks of orbitals m, so the (naux, nmo,
    norbs) intermediate of pyscf.gw.gw_ac.get_sigma is never formed at once.
  * Out of core (outcore=True, or automatically when the MO integrals do not fit), the
    occupied-virtual block and the L[:, m, orbs] block are transformed once into a temporary
    HDF5 file and the frequencies are processed in batches whose eps^-1 matrices fit in
    memory.  pyscf.gw.gw_ac.get_sigma_outcore instead keeps (I - Pi)^-1 for all nw
    frequencies in memory at once (nw * naux^2 doubles, e.g. 20 GB for nw = 100 and
    naux = 5000), which usually exceeds the in-core memory it is meant to avoid.  The
    off-diagonal (fullsigma) out-of-core path also works here (the upstream one reshapes
    with a negative dimension unless segsize = 1).

The functions keep the call signatures of their pyscf.gw.gw_ac counterparts so that they can
replace them there.
'''

import time
import numpy as np
import scipy.linalg as sla
from scipy.optimize import newton

from pyscf import lib, scf, dft
from pyscf.scf import hf
from pyscf.lib import logger, temporary_env
from pyscf.ao2mo._ao2mo import nr_e2
from pyscf.gw.gw_ac import GWAC, set_frozen_orbs, _mo_energy_without_core, _mo_without_core
from pyscf.gw.utils.ac_grid import _get_scaled_legendre_roots, PadeAC, TwoPoleAC

einsum = lib.einsum


def _mem_words(gw, reserve=0):
    '''Doubles available for temporaries: half of what is left under gw.max_memory (MB), minus
    reserve doubles, at least about 10 MB.'''
    avail = max(gw.max_memory - lib.current_memory()[0], 100) * 0.5e6 / 8
    return int(max(avail - reserve, 1.25e6))


def _orbs_range(orbs):
    '''(start, stop) if orbs is a contiguous range, else None.'''
    orbs = list(orbs)
    if orbs == list(range(orbs[0], orbs[0] + len(orbs))):
        return orbs[0], orbs[0] + len(orbs)
    return None


class _BlockSource:
    '''Row blocks L[:, rows[r0:r1], cols] of the MO three-center integrals, either as views of an
    in-core Lpq or read from a temporary HDF5 file filled once from gw.loop_ao2mo.

    rows : (start, stop) orbital range; cols : (start, stop) range or list of orbitals.
    '''

    def __init__(self, gw, rows, cols, Lpq=None, mo_coeff=None, h5file=None, name=None):
        self.rows = rows
        self.nrow = rows[1] - rows[0]
        if not isinstance(cols, tuple) and _orbs_range(cols) is not None:
            cols = _orbs_range(cols)      # basic slicing: views instead of copies in core
        if isinstance(cols, tuple):
            self.cols, self.ncol = slice(*cols), cols[1] - cols[0]
            span, sel = cols, None
        else:
            cols = list(cols)
            self.cols, self.ncol = cols, len(cols)
            rng = _orbs_range(cols)
            span = rng if rng is not None else (min(cols), max(cols) + 1)
            sel = None if rng is not None else [c - span[0] for c in cols]
        self.Lpq = Lpq
        if Lpq is not None:
            self.naux = Lpq.shape[0]
            return
        # out of core: transform once into the HDF5 file, in row segments
        with temporary_env(gw.with_df, verbose=0), temporary_env(gw.mol, verbose=0):
            naux = gw.with_df.get_naoaux()
        self.naux = naux
        nspan = span[1] - span[0]
        self.dset = h5file.create_dataset(name, (naux, self.nrow, self.ncol), 'f8')
        seg = max(1, min(self.nrow, _mem_words(gw) // max(2 * naux * nspan, 1)))
        for r0, r1 in lib.prange(0, self.nrow, seg):
            ij = (rows[0] + r0, rows[0] + r1, span[0], span[1])
            blk = gw.loop_ao2mo(mo_coeff=mo_coeff, ijslice=ij)
            if sel is not None:
                blk = blk[:, :, sel]
            self.dset[:, r0:r1] = blk
            del blk

    def block(self, r0, r1):
        '''Rows r0:r1 (relative to rows[0]) as a C-contiguous (naux, r1 - r0, ncol) array.'''
        if self.Lpq is not None:
            p0 = self.rows[0]
            return np.ascontiguousarray(self.Lpq[:, p0 + r0:p0 + r1, self.cols])
        return np.asarray(self.dset[:, r0:r1])


def get_sigma_lowmem(gw, orbs, quad_freqs, quad_wts, ef, mo_energy, mo_coeff=None, mo_occ=None,
                     iw_cutoff=None, eval_freqs=None, mo_energy_w=None, fullsigma=False, Lpq=None):
    '''GW correlation self-energy on the imaginary axis (equation 27 of
    doi.org/10.1021/acs.jctc.0c00704), with memory bounded by gw.max_memory.

    Same arguments and results as pyscf.gw.gw_ac.get_sigma (in core, Lpq given) and
    pyscf.gw.gw_ac.get_sigma_outcore (Lpq None: integrals from gw.loop_ao2mo via a temporary
    HDF5 file).

    Returns
    -------
    sigma : complex 2d (norbs, nw) or 3d (norbs, norbs, nw) array
    omega : complex 1d array, imaginary frequencies of the self-energy
    '''
    if eval_freqs is None:
        eval_freqs = quad_freqs
    log = logger.new_logger(gw)
    nocc = gw.nocc
    nmo = gw.nmo
    nw = len(quad_freqs)
    norbs = len(orbs)
    orbs = list(orbs)
    metal = hasattr(gw._scf, 'sigma')

    mo_energy_g = mo_energy
    if mo_energy_w is None:
        mo_energy_w = mo_energy
    if mo_coeff is None:
        mo_coeff = _mo_without_core(gw, gw.mo_coeff)
    if mo_occ is None:
        mo_occ = _mo_energy_without_core(gw, gw.mo_occ)

    if iw_cutoff is not None and gw.rdm is False:
        nw_sigma = sum(eval_freqs < iw_cutoff)
    else:
        nw_sigma = len(eval_freqs)
    nw_cutoff = nw_sigma if iw_cutoff is None else sum(eval_freqs < iw_cutoff)
    omega = ef + 1j * eval_freqs[:nw_sigma]
    emo = omega[None] - mo_energy_g[:, None]

    shape = (nw_sigma, norbs) if not fullsigma else (nw_sigma, norbs, norbs)
    sigma_real = np.zeros(shape)
    sigma_imag = np.zeros(shape)

    # sources of the DF blocks: response function rows (occupied-virtual, or all pairs with
    # occupation differences for smeared references) and the W contraction L[:, m, orbs]
    resp_rows, resp_cols = ((0, nmo), (0, nmo)) if metal else ((0, nocc), (nocc, nmo))
    h5 = None
    if Lpq is not None:
        resp = _BlockSource(gw, resp_rows, resp_cols, Lpq=Lpq)
        lsrc = _BlockSource(gw, (0, nmo), orbs, Lpq=Lpq)
    else:
        h5 = lib.H5TmpFile()
        log.info('GW lowmem: transforming DF integrals into a temporary file')
        resp = _BlockSource(gw, resp_rows, resp_cols, mo_coeff=mo_coeff, h5file=h5, name='resp')
        lsrc = _BlockSource(gw, (0, nmo), orbs, mo_coeff=mo_coeff, h5file=h5, name='lmn')
    naux = resp.naux
    e_r = mo_energy_w[resp_rows[0]:resp_rows[1]]
    e_c = mo_energy_w[resp_cols[0]:resp_cols[1]]
    if metal:
        f_r = mo_occ[resp_rows[0]:resp_rows[1]]
        f_c = mo_occ[resp_cols[0]:resp_cols[1]]

    # frequency batches: in core one frequency at a time (reading blocks is free); out of core
    # as many eps^-1 matrices as fit in half of the budget (each batch reads the file once)
    if Lpq is not None:
        nwb = 1
    else:
        nwb = max(1, min(nw, _mem_words(gw) // 2 // (naux * naux)))
    nbatch = (nw + nwb - 1) // nwb
    log.info('GW lowmem: %d imaginary frequencies in %d batch(es)', nw, nbatch)

    for w0, w1 in lib.prange(0, nw, nwb):
        nb = w1 - w0
        Pi = np.zeros((nb, naux, naux))
        words = _mem_words(gw)
        rblk = max(1, min(resp.nrow, words // max(2 * naux * resp.ncol, 1)))
        for r0, r1 in lib.prange(0, resp.nrow, rblk):
            L = resp.block(r0, r1).reshape(naux, -1)
            eia = (e_r[r0:r1, None] - e_c[None, :]).ravel()
            if metal:
                fia = (f_r[r0:r1, None] - f_c[None, :]).ravel()
            for k in range(nb):
                if metal:
                    f = eia * fia / (quad_freqs[w0 + k]**2 + eia**2)
                else:
                    f = 4.0 * eia / (quad_freqs[w0 + k]**2 + eia**2)
                Pi[k] += (L * f) @ L.T
            del L
        # Pi[k] <- (I - Pi)^-1 - I, in place (Pi is symmetric: invert the Fortran-ordered view)
        for k in range(nb):
            Pi[k] *= -1.0
            Pi[k][np.diag_indices(naux)] += 1.0
            Pi[k] = sla.inv(Pi[k].T, overwrite_a=True, check_finite=False)
            Pi[k][np.diag_indices(naux)] -= 1.0

        words = _mem_words(gw)
        per_m = naux * norbs * 3 + (norbs * norbs if fullsigma else 0)
        mblk = max(1, min(nmo, words // max(per_m, 1)))
        for m0, m1 in lib.prange(0, nmo, mblk):
            lm = lsrc.block(m0, m1)                        # (naux, m1 - m0, norbs)
            l2 = lm.reshape(naux, -1)
            for k in range(nb):
                w = w0 + k
                g0 = quad_wts[w] * emo[m0:m1] / (emo[m0:m1]**2 + quad_freqs[w]**2)   # (m, nw_sigma)
                if not fullsigma:
                    Q = Pi[k] @ l2
                    Q *= l2
                    Wmn = Q.sum(axis=0).reshape(m1 - m0, norbs)
                    del Q
                    sigma_real -= g0.real.T @ Wmn / np.pi
                    sigma_imag -= g0.imag.T @ Wmn / np.pi
                else:
                    Wmn = np.empty((m1 - m0, norbs, norbs))
                    for j in range(m1 - m0):
                        Wmn[j] = lm[:, j, :].T @ (Pi[k] @ lm[:, j, :])
                    Wmn = Wmn.reshape(m1 - m0, -1)
                    sigma_real.reshape(nw_sigma, -1)[:] -= g0.real.T @ Wmn / np.pi
                    sigma_imag.reshape(nw_sigma, -1)[:] -= g0.imag.T @ Wmn / np.pi
            del lm, l2
        del Pi
    if h5 is not None:
        h5.close()

    sigma = sigma_real + 1.0j * sigma_imag
    if not fullsigma:
        sigma = np.ascontiguousarray(sigma.transpose(1, 0))
    else:
        sigma = np.ascontiguousarray(sigma.transpose(1, 2, 0))
    if gw.rdm is True:
        gw.sigmaI = sigma
    return sigma[..., :nw_cutoff], omega[:nw_cutoff]


def get_sigma_exchange_direct(gw, mo_coeff):
    '''Exchange self-energy (EXX) in the MO basis, as GWAC.get_sigma_exchange.

    For a Kohn-Sham reference, upstream builds a fresh conventional RHF object whose get_veff
    stores the four-index ERIs in core whenever they fit into max_memory (nao^4/8 doubles).
    Here the exchange matrix of such references is built integral-direct (same integrals and
    result, no ERI storage).  Hartree-Fock references use their own get_veff as upstream.
    '''
    mf = gw._scf
    dm = mf.make_rdm1()
    if (not isinstance(mf, dft.rks.RKS)) and isinstance(mf, scf.hf.RHF):
        vk_ao = mf.get_veff(dm=dm) - mf.get_j(dm=dm)
    else:
        rhf = scf.RHF(gw.mol)
        vk_ao = -0.5 * hf.SCF.get_jk(rhf, gw.mol, dm, hermi=1, with_j=False)[1]
    return mo_coeff.T @ vk_ao @ mo_coeff


def hf_veff_j(gw, dm):
    '''Hartree-Fock potential J - K/2 and J (AO basis) of density matrix dm, with the same
    choice of integrals as pyscf.gw.evgw.kernel (the reference's own get_veff for HF
    references, otherwise conventional integrals) but integral-direct instead of in-core
    four-index ERIs for Kohn-Sham references.'''
    mf = gw._scf
    if (not isinstance(mf, dft.rks.RKS)) and isinstance(mf, scf.hf.RHF):
        return mf.get_veff(dm=dm), mf.get_j(dm=dm)
    vj, vk = hf.SCF.get_jk(scf.RHF(gw.mol), gw.mol, dm, hermi=1)
    return vj - 0.5 * vk, vj


def veff_lowmem(rhf, dm):
    '''rhf.get_veff(dm), but integral-direct when rhf would otherwise build in-core four-index
    ERIs (no density fitting and no _eri yet).'''
    if getattr(rhf, 'with_df', None) is not None or getattr(rhf, '_eri', None) is not None:
        return rhf.get_veff(dm=dm)
    vj, vk = hf.SCF.get_jk(rhf, rhf.mol, dm, hermi=1)
    return vj - 0.5 * vk


def kernel(gw):
    '''G0W0-AC kernel; pyscf.gw.gw_ac.kernel with the self-energy from get_sigma_lowmem.'''
    mf = gw._scf
    nocc = gw.nocc

    set_frozen_orbs(gw)
    orbs = gw.orbs
    orbs_frz = gw.orbs_frz

    mo_energy_frz = _mo_energy_without_core(gw, gw.mo_energy)
    mo_coeff_frz = _mo_without_core(gw, gw.mo_coeff)

    if gw.Lpq is None and not gw.outcore:
        with temporary_env(gw.with_df, verbose=0), temporary_env(gw.mol, verbose=0):
            gw.Lpq = gw.ao2mo(mo_coeff_frz)      # may switch gw.outcore on instead
    Lpq = None if gw.outcore else gw.Lpq

    with temporary_env(mf, verbose=0):
        dm0 = mf.make_rdm1()
        v_mf = mo_coeff_frz.T @ (mf.get_veff(dm=dm0) - mf.get_j(dm=dm0)) @ mo_coeff_frz
    gw.vxc = v_mf

    if gw.vhf_df and Lpq is not None:
        vk = -einsum('Lpi,Liq->pq', Lpq[:, :, :nocc], Lpq[:, :nocc, :])
    else:
        vk = gw.get_sigma_exchange(mo_coeff=mo_coeff_frz)
    gw.vk = vk

    gw.ef = ef = gw.get_ef(mo_energy=mf.mo_energy)

    quad_freqs, quad_wts = _get_scaled_legendre_roots(gw.nw)
    eval_freqs_with_zero = gw.setup_evaluation_grid(fallback_freqs=quad_freqs, fallback_wts=quad_wts)

    sigmaI, omega = get_sigma_lowmem(
        gw, orbs_frz, quad_freqs, quad_wts, ef, mo_energy_frz, mo_coeff_frz, iw_cutoff=gw.ac_iw_cutoff,
        eval_freqs=eval_freqs_with_zero, fullsigma=gw.fullsigma, Lpq=Lpq)

    if gw.ac == 'twopole':
        acobj = TwoPoleAC(orbs_frz, nocc)
    elif gw.ac == 'pade':
        acobj = PadeAC(npts=gw.ac_pade_npts, step_ratio=gw.ac_pade_step_ratio)
        if gw.ac_idx is not None:
            acobj.idx = gw.ac_idx
    else:
        raise ValueError('Unknown analytic continuation %s' % gw.ac)
    acobj.ac_fit(sigmaI, omega, axis=-1)
    diag_acobj = acobj.diagonal(axis1=0, axis2=1) if gw.fullsigma else acobj

    mo_energy = np.zeros_like(gw._scf.mo_energy)
    for ip, (p_in_frz, p_in_all) in enumerate(zip(orbs_frz, orbs)):
        e0 = gw._scf.mo_energy[p_in_all]
        vxx = vk[p_in_frz, p_in_frz] - v_mf[p_in_frz, p_in_frz]
        if gw.qpe_linearized is True:
            de = 1e-6
            sigmaR = diag_acobj[ip].ac_eval(e0).real
            dsigma = diag_acobj[ip].ac_eval(e0 + de).real - sigmaR
            zn = 1.0 / (1.0 - dsigma / de)
            if gw.qpe_linearized_range is not None:
                zn = 1.0 if zn < gw.qpe_linearized_range[0] or zn > gw.qpe_linearized_range[1] else zn
            mo_energy[p_in_all] = e0 + zn * (sigmaR + vxx)
        else:
            def quasiparticle(omega):
                return omega - e0 - (diag_acobj[ip].ac_eval(omega).real + vxx)
            try:
                mo_energy[p_in_all] = newton(quasiparticle, e0, tol=gw.qpe_tol, maxiter=gw.qpe_max_iter)
            except RuntimeError:
                logger.warn(gw, 'QPE for orbital=%d not converged!', p_in_all)

    gw.acobj = acobj
    gw.mo_energy = mo_energy
    with np.printoptions(threshold=len(mo_energy)):
        logger.debug(gw, '  GW mo_energy =\n%s', mo_energy)
    if gw.writefile > 0:
        logger.warn(gw, 'GWACLowMem: writefile is not supported, nothing written')
    return


class GWACLowMem(GWAC):
    '''G0W0 with analytic continuation and memory bounded by max_memory.

    Same attributes and results as pyscf.gw.gw_ac.GWAC.  If the MO three-center integrals
    do not fit into max_memory, the calculation switches to the out-of-core path instead of
    raising MemoryError.  The exchange self-energy of Kohn-Sham references is built
    integral-direct instead of from in-core four-index ERIs.
    '''

    def get_sigma_exchange(self, mo_coeff):
        return get_sigma_exchange_direct(self, mo_coeff)

    def ao2mo(self, mo_coeff=None):
        '''In-core MO integrals (naux, nmo, nmo), or None after switching to outcore when they
        do not fit into max_memory.'''
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        nmo = mo_coeff.shape[1]
        naux = self.with_df.get_naoaux()
        need = naux * nmo * nmo * 8 / 1e6
        if need + lib.current_memory()[0] < self.max_memory or self.mol.incore_anyway:
            # (upstream GWAC.ao2mo checks for twice this amount)
            mo = np.asarray(mo_coeff, order='F')
            Lpq = nr_e2(self.with_df._cderi, mo, (0, nmo, 0, nmo), aosym='s2')
            return Lpq.reshape(naux, nmo, nmo)
        logger.info(self, 'MO integrals (%.0f MB) do not fit into max_memory; switching to outcore', need)
        self.outcore = True
        return None

    def kernel(self):
        '''Do one-shot GW calculation using analytical continuation.'''
        if hasattr(self._scf, 'sigma'):
            assert self.frozen is None or self.nmo == self.mo_occ.size
            self.nw = max(400, self.nw)
            self.ac_pade_npts = 18
            self.ac_pade_step_ratio = 5.0 / 6.0
        if self.Lpq is None:
            self.initialize_df(auxbasis=self.auxbasis)
        cput0 = (time.process_time(), time.perf_counter())
        self.dump_flags()
        kernel(self)
        logger.timer(self, 'GW (lowmem)', *cput0)
        return
