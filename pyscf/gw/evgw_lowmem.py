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
Low-memory variant of spin-restricted evGW / evGW0 with analytic continuation
(pyscf.gw.evgw.EVGW).

Same algorithm and results as pyscf.gw.evgw.EVGW (to rounding), with

  * the imaginary-axis self-energy of every cycle from
    pyscf.gw.gw_ac_lowmem.get_sigma_lowmem (blocked response function and W contraction;
    out of core: DF integrals transformed once into a temporary file, frequency batches
    instead of nw * naux^2 doubles), and an automatic switch to out of core when the MO
    integrals do not fit into max_memory;
  * the Hartree-Fock Hamiltonian of Kohn-Sham references built integral-direct instead of
    from in-core four-index ERIs;
  * a fix for a preset gw.Lpq (pyscf.gw.evgw.kernel only binds its local Lpq when it
    transforms the integrals itself, so a GW object with Lpq already set fails with a
    NameError in the in-core path).
'''

import time
import numpy as np
import scipy

from pyscf.lib import diis, einsum, logger, temporary_env
from pyscf.gw.evgw import EVGW
from pyscf.gw.gw_ac import set_frozen_orbs, _mo_energy_without_core, _mo_without_core
from pyscf.gw.gw_ac_lowmem import GWACLowMem, get_sigma_lowmem, hf_veff_j
from pyscf.gw.utils.ac_grid import _get_scaled_legendre_roots, PadeAC, TwoPoleAC


def kernel(gw):
    '''evGW kernel (see pyscf.gw.evgw.kernel).'''
    mf = gw._scf
    nmo = gw.nmo
    nocc = gw.nocc
    mo_energy = gw.mo_energy

    set_frozen_orbs(gw)
    orbs = gw.orbs
    orbs_frz = gw.orbs_frz

    mo_energy_frz = _mo_energy_without_core(gw, gw.mo_energy)
    mf_mo_energy_frz = _mo_energy_without_core(gw, gw._scf.mo_energy)
    mo_coeff_frz = _mo_without_core(gw, gw.mo_coeff)

    if gw.Lpq is None and not gw.outcore:
        with temporary_env(gw.with_df, verbose=0), temporary_env(gw.mol, verbose=0):
            gw.Lpq = gw.ao2mo(mo_coeff_frz)      # may switch gw.outcore on instead
    Lpq = None if gw.outcore else gw.Lpq

    hcore = mo_coeff_frz.T @ mf.get_hcore() @ mo_coeff_frz
    if gw.vhf_df is False or Lpq is None:
        if gw.vhf_df:
            logger.warn(gw, 'vhf_df needs the in-core MO integrals; using conventional J/K')
        veff, vj = hf_veff_j(gw, mf.make_rdm1())
        vjk = mo_coeff_frz.T @ veff @ mo_coeff_frz
        vk = vjk - mo_coeff_frz.T @ vj @ mo_coeff_frz
    else:
        vj = 2.0 * einsum('Lii,Lpq->pq', Lpq[:, :nocc, :nocc], Lpq)
        vk = -einsum('Lpi,Liq->pq', Lpq[:, :, :nocc], Lpq[:, :nocc, :])
        vjk = vj + vk
    gw.vk = vk
    ham_hf = hcore + vjk

    gw_diis = diis.DIIS(gw, gw.diis_file)
    gw_diis.space = gw.diis_space

    gw.ef = ef = gw.get_ef(mo_energy=mf.mo_energy)

    quad_freqs, quad_wts = _get_scaled_legendre_roots(gw.nw)
    eval_freqs_with_zero = gw.setup_evaluation_grid(fallback_freqs=quad_freqs, fallback_wts=quad_wts)

    conv = False
    cycle = 0
    while conv is False and cycle < max(1, gw.max_cycle):
        mo_energy_prev = mo_energy.copy()
        mo_energy_frz_prev = mo_energy_frz.copy()

        mo_energy_w = mf_mo_energy_frz if gw.W0 is True else mo_energy_frz
        sigmaI, omega = get_sigma_lowmem(
            gw, orbs_frz, quad_freqs, quad_wts, ef, mo_energy_frz, mo_coeff_frz, iw_cutoff=gw.ac_iw_cutoff,
            mo_energy_w=mo_energy_w, eval_freqs=eval_freqs_with_zero, fullsigma=False, Lpq=Lpq)

        if gw.ac == 'twopole':
            acobj = TwoPoleAC(orbs_frz, nocc)
        elif gw.ac == 'pade':
            acobj = PadeAC(npts=gw.ac_pade_npts, step_ratio=gw.ac_pade_step_ratio)
        else:
            raise ValueError('Unknown GW-AC type %s' % (str(gw.ac)))
        acobj.ac_fit(sigmaI, omega, axis=-1)
        gw.acobj = acobj

        mo_energy = mo_energy_prev.copy()
        for ip, p in enumerate(orbs_frz):

            def quasiparticle(omega):
                return omega - (ham_hf[p, p] + acobj[ip].ac_eval(omega).real)

            try:
                mo_energy[orbs[ip]] = scipy.optimize.newton(
                    quasiparticle, mo_energy_prev[orbs[ip]], tol=gw.qpe_tol, maxiter=gw.qpe_max_iter)
            except RuntimeError:
                logger.warn(gw, 'QPE for orbital=%d not converged!', orbs[ip])

        # (as upstream, the next cycle's Newton guesses are the QP energies before DIIS)
        gw.mo_energy = gw_diis.update(mo_energy)
        mo_energy_frz = _mo_energy_without_core(gw, gw.mo_energy)
        gw.ef = ef = gw.get_ef(mo_energy=gw.mo_energy)

        diff = abs(np.sum(1.0 / mo_energy_frz - 1.0 / mo_energy_frz_prev)) / nmo / nmo
        if diff < gw.conv_tol:
            conv = True
        logger.info(gw, 'EVGW cycle= %d  |delta G|= %4.3g', cycle + 1, diff)
        cycle += 1
        with np.printoptions(threshold=len(gw._scf.mo_energy)):
            logger.debug(gw, '  GW mo_energy =\n%s', gw.mo_energy)

    gw.converged = conv
    gw.cycle = cycle
    logger.debug(gw, 'EVGW %s in %-d cycles.', 'converged' if conv else 'not converged', cycle)
    return


class EVGWLowMem(EVGW):
    '''evGW / evGW0 with analytic continuation and memory bounded by max_memory.

    Same attributes and results as pyscf.gw.evgw.EVGW.
    '''

    ao2mo = GWACLowMem.ao2mo
    get_sigma_exchange = GWACLowMem.get_sigma_exchange

    def kernel(self):
        '''Run an eigenvalue-self-consistent GW calculation.'''
        if hasattr(self._scf, 'sigma'):
            assert self.frozen == 0 or self.frozen is None
            self.nw = max(400, self.nw)
            self.ac_pade_npts = 18
            self.ac_pade_step_ratio = 5.0 / 6.0
        if self.Lpq is None:
            self.initialize_df(auxbasis=self.auxbasis)
        cput0 = (time.process_time(), time.perf_counter())
        self.dump_flags()
        kernel(self)
        logger.timer(self, 'evGW (lowmem)', *cput0)
        return
