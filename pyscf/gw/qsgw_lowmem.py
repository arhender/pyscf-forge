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
Low-memory variant of spin-restricted quasiparticle self-consistent GW with analytic
continuation (pyscf.gw.qsgw.QSGW).

Same algorithm and results as pyscf.gw.qsgw.QSGW (to rounding) for frozen=None, with

  * the full (off-diagonal) imaginary-axis self-energy of every cycle from
    pyscf.gw.gw_ac_lowmem.get_sigma_lowmem: W_mnn' is accumulated over blocks of m
    (pyscf.gw.gw_ac.get_sigma allocates it for all nmo at once, nmo * norbs^2 doubles), the
    response function over occupied blocks, and out of core the frequencies are batched
    instead of keeping nw * naux^2 doubles; the calculation switches to out of core when the
    MO integrals do not fit into max_memory;
  * the previous cycle's MO integrals released before the new ones are transformed
    (upstream holds both at the peak);
  * the Hartree-Fock potential built integral-direct when the HF object has neither density
    fitting nor stored ERIs (instead of storing the four-index ERIs in core).

Optional, experimental (scissor='hf' or 'mf'): with a GW window, orbitals outside it get a
diagonal static self-energy derived from the nearest window orbital of the same kind instead
of none (upstream leaves them as Hartree-Fock orbitals); see QSGWLowMem.scissor.

The two-pole fit gets the window's orbital indices for its occupied/virtual initial guesses
(upstream passes range(nmo), which only matches without a window).

With frozen orbitals, upstream transforms the integrals with all orbitals in core but uses
them with the frozen-space orbital indices (the out-of-core path uses the frozen-space
orbitals); here both paths use the frozen-space orbitals.  With vhf_df and frozen orbitals
(where the MO-basis formula of upstream would need the frozen orbitals too) J and K are
taken from the density-fitting object in the AO basis.
'''

import h5py
import numpy as np
import scipy

from pyscf import scf
from pyscf.lib import einsum, logger, temporary_env
from pyscf.gw.qsgw import QSGW
from pyscf.gw.gw_ac import set_frozen_orbs, _mo_energy_without_core, _mo_without_core
from pyscf.gw.gw_ac_lowmem import GWACLowMem, get_sigma_lowmem, veff_lowmem
from pyscf.gw.utils.ac_grid import _get_scaled_legendre_roots, PadeAC, TwoPoleAC


def kernel(gw):
    '''qsGW kernel (see pyscf.gw.qsgw.kernel).'''
    mf = gw._scf
    nmo = gw.nmo
    nocc = gw.nocc

    if gw.load_chkfile:
        with h5py.File(gw.chkfile, 'r') as fh5:
            gw.mo_energy = np.array(fh5['gw/mo_energy'])
            gw.mo_coeff = np.array(fh5['gw/mo_coeff'])

    set_frozen_orbs(gw)
    orbs_frz = gw.orbs_frz
    frozen = gw.frozen is not None and np.size(gw.frozen) > 0 and np.any(np.asarray(gw.frozen) != 0)

    mo_energy_frz = _mo_energy_without_core(gw, gw.mo_energy)
    mo_coeff_frz = _mo_without_core(gw, gw.mo_coeff)
    mo_occ_frz = _mo_energy_without_core(gw, gw.mo_occ)

    quad_freqs, quad_wts = _get_scaled_legendre_roots(gw.nw)
    eval_freqs_with_zero = gw.setup_evaluation_grid(fallback_freqs=quad_freqs, fallback_wts=quad_wts)

    hcore = mf.get_hcore()
    ovlp = mf.get_ovlp()

    gw.hf_obj = rhf = gw.set_hf_object()
    dm_iter = rhf.make_rdm1(mo_energy=gw.mo_energy, mo_coeff=gw.mo_coeff)

    gw_diis = scf.diis.DIIS(gw, gw.diis_file)
    gw_diis.space = gw.diis_space

    gw.ef = ef = gw.get_ef(mo_energy=gw.mo_energy)

    v_mf_minus_hf = None
    if gw.scissor == 'mf':
        # v_xc(reference) - v_x(HF) at the reference density (AO basis); zero for HF references
        dm0 = mf.make_rdm1()
        with temporary_env(mf, verbose=0):
            v_mf_minus_hf = (mf.get_veff(dm=dm0) - mf.get_j(dm=dm0)) - (veff_lowmem(rhf, dm0) - rhf.get_j(dm=dm0))

    conv = False
    cycle = 0
    while conv is False and cycle < max(1, gw.max_cycle):
        # MO integrals for the current orbitals (previous ones released first)
        gw.Lpq = None
        if gw.Lpq_ao is not None:
            gw.Lpq = einsum('Lmn,mp,nq->Lpq', gw.Lpq_ao, mo_coeff_frz, mo_coeff_frz)
        elif not gw.outcore:
            with temporary_env(gw.with_df, verbose=0), temporary_env(gw.mol, verbose=0):
                gw.Lpq = gw.ao2mo(mo_coeff_frz)      # may switch gw.outcore on instead
        Lpq = gw.Lpq

        nfrz_sigma = gw.get_static_sigma(Lpq, orbs_frz, mo_energy_frz, mo_coeff_frz, mo_occ_frz, ef,
                                         quad_freqs, quad_wts, eval_freqs_with_zero)

        sigma = np.zeros(shape=[nmo, nmo], dtype=np.complex128)
        sigma[np.ix_(orbs_frz, orbs_frz)] = nfrz_sigma
        if gw.scissor not in (False, None, True, 'hf', 'mf'):
            raise ValueError("QSGW scissor must be False, 'hf' or 'mf'")
        if gw.scissor:
            dmf = None
            if gw.scissor == 'mf':
                dmf = np.diag(mo_coeff_frz.T @ v_mf_minus_hf @ mo_coeff_frz)
            scissor_sigma(sigma, orbs_frz, mo_energy_frz, nocc, log=gw, dmf=dmf)
        gw.sigma_static = sigma.real.copy()

        CS = mo_coeff_frz.T @ ovlp
        vsig = CS.T @ sigma.real @ CS

        if gw.vhf_df is False:
            veff = veff_lowmem(rhf, dm_iter)
        elif Lpq is not None and not frozen:
            vj = 2.0 * einsum('Lii,Lpq->pq', Lpq[:, :nocc, :nocc], Lpq)
            vk = -einsum('Lpi,Liq->pq', Lpq[:, :, :nocc], Lpq[:, :nocc, :])
            veff = CS.T @ (vj + vk) @ CS
        else:
            vj, vk = gw.with_df.get_jk(dm_iter, hermi=1)
            veff = vj - 0.5 * vk

        ham = hcore + veff + vsig
        ham = gw_diis.update(ovlp, dm_iter, ham)

        gw.mo_energy, gw.mo_coeff = scipy.linalg.eigh(ham, ovlp)
        gw.mo_occ = rhf.get_occ(mo_energy=gw.mo_energy, mo_coeff=gw.mo_coeff)
        gw.ef = ef = gw.get_ef()
        gw.vsig = vsig

        mo_energy_frz = _mo_energy_without_core(gw, gw.mo_energy)
        mo_coeff_frz = _mo_without_core(gw, gw.mo_coeff)
        mo_occ_frz = _mo_energy_without_core(gw, gw.mo_occ)

        dm_old = dm_iter.copy()
        dm_iter = rhf.make_rdm1(mo_coeff=gw.mo_coeff, mo_occ=gw.mo_occ)
        norm_dm = np.linalg.norm(dm_iter - dm_old) / nmo
        if norm_dm < gw.conv_tol:
            conv = True

        logger.info(gw, 'QSGW cycle= %d  |ddm|= %4.3g', cycle + 1, norm_dm)
        cycle += 1
        with np.printoptions(threshold=len(gw.mo_energy)):
            logger.debug(gw, '  GW mo_energy =\n%s', gw.mo_energy)
        if gw.chkfile:
            gw.dump_chk()

    gw.converged = conv
    gw.cycle = cycle
    logger.debug(gw, 'QSGW %s in %-d cycles.', 'converged' if conv else 'not converged', cycle)
    return


def get_static_sigma_ac(gw, Lpq, orbs_frz, mo_energy_frz, mo_coeff_frz, mo_occ_frz, ef,
                        quad_freqs, quad_wts, eval_freqs):
    '''Static, Hermitian qsGW self-energy (len(orbs), len(orbs)) of the current orbitals from the
    imaginary-axis self-energy and analytic continuation (gw.mode 'a', 'b' or 'c'; as
    pyscf.gw.qsgw.kernel).  Sets gw.acobj.'''
    nocc = gw.nocc
    eta = gw.eta
    sigmaI, omega = get_sigma_lowmem(
        gw, orbs_frz, quad_freqs, quad_wts, ef, mo_energy_frz, mo_coeff_frz, mo_occ=mo_occ_frz,
        iw_cutoff=gw.ac_iw_cutoff, eval_freqs=eval_freqs, fullsigma=True, Lpq=Lpq)

    if gw.ac == 'twopole':
        # (upstream passes range(nmo): with a window the occupied/virtual initial guess of
        # the fit for window orbital k would then be taken from orbital k instead of orbs[k])
        acobj = TwoPoleAC(list(orbs_frz), nocc)
    elif gw.ac == 'pade':
        acobj = PadeAC(npts=gw.ac_pade_npts, step_ratio=gw.ac_pade_step_ratio)
        if gw.ac_idx is not None:
            acobj.idx = gw.ac_idx
    else:
        raise ValueError('Unknown GW-AC type %s' % (str(gw.ac)))
    acobj.ac_fit(sigmaI, omega, axis=-1)
    del sigmaI

    if gw.mode == 'b':
        sigma_at_fermi = acobj.ac_eval(1j * eta + ef)
        nfrz_sigma = 0.5 * (sigma_at_fermi + sigma_at_fermi.T.conj())
        acobj_diag = acobj.diagonal()
        for ip, p in enumerate(orbs_frz):
            nfrz_sigma[ip, ip] = acobj_diag[ip].ac_eval(mo_energy_frz[p] + 1j * eta).real
    elif gw.mode == 'a':
        nfrz_sigma = np.zeros(shape=[len(orbs_frz), len(orbs_frz)], dtype=np.complex128)
        for ip, p in enumerate(orbs_frz):
            for iq, q in enumerate(orbs_frz):
                nfrz_sigma[ip, iq] += 0.25 * acobj[ip, iq].ac_eval(mo_energy_frz[p] + 1j * eta)
                nfrz_sigma[ip, iq] += 0.25 * acobj[iq, ip].ac_eval(mo_energy_frz[p] + 1j * eta).conj()
                nfrz_sigma[ip, iq] += 0.25 * acobj[ip, iq].ac_eval(mo_energy_frz[q] + 1j * eta)
                nfrz_sigma[ip, iq] += 0.25 * acobj[iq, ip].ac_eval(mo_energy_frz[q] + 1j * eta).conj()
    elif gw.mode == 'c':
        sigma_at_fermi = acobj.ac_eval(1j * eta + ef)
        nfrz_sigma = 0.5 * (sigma_at_fermi + sigma_at_fermi.T.conj())
    else:
        raise ValueError('Unknown QSGW mode %s' % gw.mode)
    gw.acobj = acobj
    return nfrz_sigma

def scissor_sigma(sigma, orbs, mo_energy, nocc, log=None, dmf=None):
    '''Scissor-like static self-energy outside the GW window, in place.

    Every orbital p not in orbs (frozen-space indices) gets the diagonal element sigma_qq of
    the energetically nearest window orbital q of the same kind (occupied: p, q < nocc;
    virtual: p, q >= nocc), i.e. the same correlation shift relative to Hartree-Fock as q;
    with dmf (diagonal of v_xc(reference) - v_x(HF) in the same basis) additionally
    dmf[p] - dmf[q], i.e. the mean-field level of p plus the shift of q.  Elements coupling
    p to other orbitals stay zero.  Orbitals without a window orbital of their kind are left
    unchanged.
    '''
    nmo = sigma.shape[0]
    orbs = list(orbs)
    win = set(orbs)
    occ_w = np.array([q for q in orbs if q < nocc], dtype=int)
    vir_w = np.array([q for q in orbs if q >= nocc], dtype=int)
    for p in range(nmo):
        if p in win:
            continue
        cand = occ_w if p < nocc else vir_w
        if cand.size == 0:
            if log is not None:
                logger.warn(log, 'qsGW scissor: no %s window orbital for orbital %d',
                            'occupied' if p < nocc else 'virtual', p)
            continue
        q = cand[np.argmin(np.abs(mo_energy[cand] - mo_energy[p]))]
        sigma[p, p] = sigma[q, q].real
        if dmf is not None:
            sigma[p, p] += dmf[p] - dmf[q]
    return sigma


class QSGWLowMem(QSGW):
    '''qsGW with analytic continuation and memory bounded by max_memory.

    Same attributes and results as pyscf.gw.qsgw.QSGW (frozen orbitals: see module docstring).

    Additional attributes:
        scissor : False, 'hf' (or True) or 'mf'
            Experimental.  With a GW window (gw.orbs a subset of the non-frozen orbitals),
            orbitals outside the window otherwise get no correlation self-energy (they are
            Hartree-Fock orbitals in H_HF + Sigma) but enter the screening of the next cycle.
            'hf': each gets the diagonal static self-energy sigma_qq of the energetically
            nearest window orbital q of the same kind (occupied/virtual), i.e. the same
            correction relative to Hartree-Fock.  'mf': sigma_qq + d_pp - d_qq with
            d = v_xc(reference) - v_x(HF) at the reference density, i.e. the mean-field level
            plus the shift of q, as the scissor of evGW (equal to 'hf' for HF references).
            No effect without a window.  Default False (as upstream).
            Neither variant systematically reduces the window error with respect to qsGW
            over all orbitals (H2O, 6-31G and def2-SVP, windows of 4 and 7 orbitals: errors
            of 0.15-0.43 eV with or without, mean changes of +-10 meV): that error comes
            mostly from the missing window/outside off-diagonal elements of Sigma, which
            relax the window orbitals in full qsGW.

    Saved results (in addition to QSGW):
        sigma_static : static self-energy of the last cycle in the MO basis it was built in
            (non-frozen space), including the scissor elements.
    '''

    ao2mo = GWACLowMem.ao2mo
    get_sigma_exchange = GWACLowMem.get_sigma_exchange

    def __init__(self, mf, frozen=None, auxbasis=None):
        QSGW.__init__(self, mf, frozen=frozen, auxbasis=auxbasis)
        self.scissor = False
        self.sigma_static = None

    def get_static_sigma(self, Lpq, orbs_frz, mo_energy_frz, mo_coeff_frz, mo_occ_frz, ef,
                         quad_freqs, quad_wts, eval_freqs):
        '''Static qsGW self-energy of the current orbitals (window block); see get_static_sigma_ac.'''
        return get_static_sigma_ac(self, Lpq, orbs_frz, mo_energy_frz, mo_coeff_frz, mo_occ_frz, ef,
                                   quad_freqs, quad_wts, eval_freqs)

    def dump_flags(self):
        QSGW.dump_flags(self)
        logger.info(self, 'scissor outside the GW window = %s', self.scissor)

    def kernel(self):
        '''Run a quasiparticle self-consistent GW calculation.'''
        if hasattr(self._scf, 'sigma'):
            assert self.frozen == 0 or self.frozen is None
            self.nw = max(400, self.nw)
            self.ac_pade_npts = 18
            self.ac_pade_step_ratio = 5.0 / 6.0
        if self.Lpq is None:
            self.initialize_df(auxbasis=self.auxbasis)
        cput0 = (logger.process_clock(), logger.perf_counter())
        self.dump_flags()
        kernel(self)
        logger.timer(self, 'qsGW (lowmem)', *cput0)
        return
