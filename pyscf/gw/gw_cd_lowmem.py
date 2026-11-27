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
Low-memory variant of spin-restricted G0W0 with contour deformation (pyscf.gw.gw_cd).

Same theory and results as pyscf.gw.gw_cd.GWCD where that is correct; differences:

  * Memory: W_mn on the imaginary axis and the residue terms use the response function
    accumulated over blocks of occupied orbitals (no L_ia copy or L_ia-sized temporaries)
    and W is contracted over blocks of orbitals (pyscf.gw.gw_fscd.get_W_diag), so apart
    from the DF integrals Lpq only a few naux x naux matrices and max_memory-bounded blocks
    are held.  get_rho_response_R of pyscf.gw.gw_cd needs about eleven naux x naux real
    matrices per residue plus two L_ia-sized temporaries.
  * The exchange self-energy of Kohn-Sham references is built integral-direct
    (pyscf.gw.gw_ac.GWAC.get_sigma_exchange stores the four-index ERIs in core when they
    fit into max_memory).
  * Fixes for gw.orbs / gw.frozen (pyscf.gw.gw_cd.kernel gives wrong results or crashes
    unless orbs = range(n) and frozen is None):
      - W_mn on the imaginary axis is indexed by the position of the orbital in gw.orbs
        (upstream indexes it by the frozen-space orbital index, an IndexError for
        non-contiguous orbs and the wrong orbital for orbs that do not start at 0);
      - W_mn is computed for the frozen-space orbital list (upstream passes the full-space
        gw.orbs to the frozen-space integrals);
      - the Newton starting guess uses the orbital's own mean-field energy (upstream uses
        mf.mo_energy[p_frz], i.e. a different orbital when orbitals are frozen).
'''

import time
import numpy as np
import h5py
import scipy.linalg as sla
from scipy.optimize import newton

from pyscf import lib
from pyscf.lib import logger, temporary_env
from pyscf.gw.gw_ac import set_frozen_orbs, _mo_energy_without_core, _mo_without_core
from pyscf.gw.gw_cd import GWCD
from pyscf.gw.gw_fscd import get_W_diag, _dielectric
from pyscf.gw.gw_ac_lowmem import get_sigma_exchange_direct
from pyscf.gw.utils.ac_grid import _get_scaled_legendre_roots

einsum = lib.einsum


def get_sigmaR_diag(mo_energy, omega, orbp, ef, Lpq, nocc, eta, max_memory=4000):
    '''Residue part of the self-energy of orbital orbp (frozen-space index) at omega; same as
    pyscf.gw.gw_cd.get_sigmaR_diag (one dielectric matrix per residue), with the response
    function accumulated over occupied blocks.'''
    if omega > ef:
        fm = 1.0
        idx = np.where((mo_energy < omega) & (mo_energy > ef))[0]
    else:
        fm = -1.0
        idx = np.where((mo_energy > omega) & (mo_energy < ef))[0]
    naux = Lpq.shape[0]
    words = int(max(max_memory - lib.current_memory()[0], 100) * 0.5e6 / 8) - 4 * naux * naux
    sigmaR = 0j
    for m in idx:
        eps = _dielectric(Lpq, nocc, mo_energy, abs(mo_energy[m] - omega), eta, max(words, 1))
        lpm = Lpq[:, orbp, m]
        vec = sla.solve(eps.T, lpm, check_finite=False, overwrite_a=True, assume_a='sym')
        vec -= lpm
        sigmaR += fm * np.dot(Lpq[:, m, orbp], vec)
        del eps
    return sigmaR


def get_sigma_diag(ef, ep, p, mo_energy, Lpq, nocc, Wmn, freqs, wts, eta, max_memory=4000):
    '''Sigma^c_pp(ep) by contour deformation: imaginary-axis integral with Wmn (nw, nmo) of
    orbital p, plus residues (see pyscf.gw.gw_cd.get_sigma_diag).'''
    sign = np.sign(ef - mo_energy)
    emo = ep - 1j * eta * sign - mo_energy
    g0 = wts[None, :] * emo[:, None] / ((emo**2)[:, None] + (freqs**2)[None, :])
    sigmaI = -einsum('mw,wm', g0, Wmn) / np.pi
    return sigmaI + get_sigmaR_diag(mo_energy, ep, p, ef, Lpq, nocc, eta, max_memory)


def kernel(gw, load_chk=None):
    '''GWCD kernel (see pyscf.gw.gw_cd.kernel), low-memory and with the orbs/frozen fixes.

    Returns (conv, conv_inds, mo_energy).'''
    mf = gw._scf
    nocc = gw.nocc
    log = logger.new_logger(gw)

    set_frozen_orbs(gw)
    orbs = list(gw.orbs)
    orbs_frz = list(gw.orbs_frz)

    mo_energy_frz = _mo_energy_without_core(gw, gw.mo_energy)
    mo_coeff_frz = _mo_without_core(gw, gw.mo_coeff)
    if gw.Lpq is None:
        with temporary_env(gw.with_df, verbose=0), temporary_env(gw.mol, verbose=0):
            gw.Lpq = gw.ao2mo(mo_coeff_frz)
    Lpq = gw.Lpq
    nmo = Lpq.shape[1]

    mo_energy = np.zeros_like(mf.mo_energy)
    conv_inds = np.zeros(mo_energy.shape, dtype=int)
    if load_chk is not None:
        with h5py.File(load_chk, 'r') as f:
            g = f['gwcd']
            mo_energy = g['mo_energy'][()]
            conv_inds = g['conv_inds'][()]
        if mo_energy.shape != (mf.mo_energy.size,) or conv_inds.shape != mo_energy.shape:
            raise RuntimeError(f'{load_chk}: shape mismatch')
    chkfile = gw.chkfile
    if chkfile is not None:
        with h5py.File(chkfile, 'a') as f:
            if 'gwcd' in f:
                del f['gwcd']
            g = f.create_group('gwcd')
            g.create_dataset('mo_energy', data=mo_energy)
            g.create_dataset('conv_inds', data=conv_inds)

    with temporary_env(mf, verbose=0):
        dm0 = mf.make_rdm1()
        v_mf = mo_coeff_frz.T @ (mf.get_veff(dm=dm0) - mf.get_j(dm=dm0)) @ mo_coeff_frz
    gw.vxc = v_mf
    if gw.vhf_df:
        vk = -einsum('Lpi,Liq->pq', Lpq[:, :, :nocc], Lpq[:, :nocc, :])
    else:
        vk = gw.get_sigma_exchange(mo_coeff=mo_coeff_frz)
    gw.vk = vk
    gw.ef = gw.get_ef(mo_energy=mf.mo_energy)

    quad_freqs, quad_wts = _get_scaled_legendre_roots(gw.nw)
    log.debug('Computing Wmn on the imaginary axis (%d points)', gw.nw)
    # (nw, len(orbs), nmo), indexed by the position of the orbital in gw.orbs
    Wmn = get_W_diag(Lpq, nocc, mo_energy_frz, quad_freqs, [(orbs_frz, slice(0, nmo))],
                     max_memory=gw.max_memory)[0]

    conv = True
    for ip, (p_in_frz, p_in_all) in enumerate(zip(orbs_frz, orbs)):
        if gw.qpe_linearized:
            logger.warn(gw, 'linearization with CD leads to wrong quasiparticle energy')
            raise NotImplementedError
        if load_chk is not None and conv_inds[p_in_all]:
            continue
        e0 = mf.mo_energy[p_in_all]
        vxx = vk[p_in_frz, p_in_frz] - v_mf[p_in_frz, p_in_frz]

        def quasiparticle(omega):
            sigma = get_sigma_diag(gw.ef, omega, p_in_frz, mo_energy_frz, Lpq, nocc, Wmn[:, ip],
                                   quad_freqs, quad_wts, gw.eta, gw.max_memory).real
            return omega - e0 - (sigma + vxx)

        delta = -1e-2 if p_in_frz < nocc else 1e-2
        try:
            e, result = newton(quasiparticle, e0 + delta, tol=gw.qpe_tol, maxiter=gw.qpe_max_iter,
                               full_output=True)
            log.debug('QP energy for orb %d: %s (%d iterations)', p_in_all, e, result.iterations)
            mo_energy[p_in_all] = e
            conv_inds[p_in_all] = 1
            if chkfile is not None:
                with h5py.File(chkfile, 'r+') as f:
                    g = f['gwcd']
                    g['mo_energy'][()] = mo_energy[()]
                    g['conv_inds'][()] = conv_inds[()]
        except RuntimeError:
            conv = False
            log.warn('QP equation for orbital %d did not converge', p_in_all)

    if gw.verbose >= logger.DEBUG:
        with np.printoptions(threshold=len(mo_energy)):
            log.debug('  GW mo_energy =\n%s', mo_energy)
    return conv, conv_inds, mo_energy


class GWCDLowMem(GWCD):
    '''G0W0 with contour deformation; low-memory and correct for any gw.orbs / gw.frozen.

    Same attributes as pyscf.gw.gw_cd.GWCD (outcore is not supported).  The exchange
    self-energy of Kohn-Sham references is built integral-direct (no in-core 4-index ERIs).
    '''

    def get_sigma_exchange(self, mo_coeff):
        return get_sigma_exchange_direct(self, mo_coeff)

    def kernel(self, load_chk=None):
        '''Do one-shot GW calculation using contour deformation.'''
        if self.outcore:
            raise NotImplementedError('GWCDLowMem keeps the MO integrals in memory (outcore not supported)')
        if self.Lpq is None:
            self.initialize_df(auxbasis=self.auxbasis)
        cput0 = (time.process_time(), time.perf_counter())
        self.dump_flags()
        self.converged, self.conv_inds, self.mo_energy = kernel(self, load_chk=load_chk)
        logger.timer(self, 'GWCD (lowmem)', *cput0)
        return self.mo_energy
