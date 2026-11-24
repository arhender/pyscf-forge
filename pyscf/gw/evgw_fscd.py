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
Spin-restricted eigenvalue self-consistent GW (evGW, evGW0) with frequency-sampled contour
deformation (fsCD-evGW).

In every cycle n the quasiparticle equation

    e_p^(n+1) = e_p^MF + Sigma^x_pp - vxc_pp + Re Sigma^c_pp(e_p^(n+1); {e^(n)})

is solved with fsCD (pyscf.gw.gw_fscd) for the orbitals in gw.orbs (the "window"), where the
self-energy is built from the previous quasiparticle spectrum {e^(n)}: in the Green's function
(residue positions, imaginary-axis G) and, for evGW, in the screening (response function of W).
evGW0 keeps W at the mean-field energies.

Windowed calculations: orbitals outside gw.orbs are not solved explicitly.  After each cycle
they are scissor-shifted (pyscf.gw.qp_shift.scissor_shift_energies): every occupied (virtual)
orbital outside the window receives the correction of the computed occupied (virtual) orbital
closest in mean-field energy.  The complete, shifted spectrum enters both G and the
screening of the next cycle, so orbitals outside the window are not left at their mean-field
energies in the response function.  Frozen orbitals (gw.frozen) are excluded from GW altogether and keep their
mean-field energies.

Ref: M. Kehry, W. Klopper, C. Holzer, J. Chem. Phys. 159, 044116 (2023)
'''

import time
import numpy as np

from pyscf.lib import logger, diis
from pyscf.gw.gw_ac import get_frozen_mask
from pyscf.gw.gw_fscd import GWFSCD, prepare, setup, solve_qp
from pyscf.gw.qp_shift import scissor_shift_energies


def kernel(gw):
    '''fsCD-evGW kernel.

    Returns (conv, conv_inds, mo_energy): conv is True if the self-consistency and all QP
    equations of the last cycle converged; mo_energy contains all orbitals (window orbitals
    from GW, the others scissor-shifted, frozen orbitals at their mean-field energies).
    '''
    mf = gw._scf
    log = logger.new_logger(gw)
    ints = prepare(gw)
    orbs = list(gw.orbs)
    orbs_frz = ints['orbs_frz']
    mask = get_frozen_mask(gw)
    e_mf_frz = ints['mo_energy_mf']
    occ_frz = np.asarray(mf.mo_occ)[mask]

    # current spectrum: full MO space and non-frozen subspace
    mo_energy = np.array(mf.mo_energy, dtype=float)
    e_frz = e_mf_frz.copy()

    gw_diis = None
    if gw.diis_space and gw.diis_space > 0:
        gw_diis = diis.DIIS(gw, gw.diis_file)
        gw_diis.space = gw.diis_space

    conv = False
    conv_inds = np.zeros(mo_energy.shape, dtype=int)
    guess = None
    for cycle in range(max(1, gw.max_cycle)):
        ef = gw.get_ef(mo_energy=mo_energy)
        e_w = e_mf_frz if gw.W0 else e_frz
        setup(gw, mo_energy_g=e_frz, mo_energy_w=e_w, ef=ef)
        qp_conv, conv_inds, qp = solve_qp(gw, guess=guess)

        win_old = mo_energy[orbs]
        win_new = qp[orbs]
        failed = conv_inds[orbs] == 0
        if failed.any():
            # keep the previous value of orbitals whose QP equation did not converge
            win_new = np.where(failed, win_old, win_new)
        if gw_diis is not None and cycle >= gw.diis_start_cycle:
            win_new = gw_diis.update(win_new)

        # scissor shift of everything outside the window (non-frozen space)
        qp_frz = e_mf_frz.copy()
        qp_frz[orbs_frz] = win_new
        e_frz = scissor_shift_energies(qp_frz, e_mf_frz, occ_frz, orbs_frz,
                                       log=log if log.verbose >= logger.DEBUG else None)
        mo_energy = np.array(mf.mo_energy, dtype=float)
        mo_energy[mask] = e_frz

        diff = np.abs(win_new - win_old).max()
        log.info('fsCD-%s cycle %d  max|dE| = %.3e Ha  (real samples %d)', 'evGW0' if gw.W0 else 'evGW',
                 cycle + 1, diff, gw._fs['samples'].freqs.size)
        guess = mo_energy
        if diff < gw.conv_tol and qp_conv and not failed.any():
            conv = True
            break

    gw.cycle = cycle + 1
    gw.real_freqs = gw._fs['samples'].freqs
    if conv:
        log.info('fsCD-%s converged in %d cycles', 'evGW0' if gw.W0 else 'evGW', gw.cycle)
    else:
        log.warn('fsCD-%s not converged in %d cycles', 'evGW0' if gw.W0 else 'evGW', gw.cycle)
    return conv, conv_inds, mo_energy


class EVGWFSCD(GWFSCD):
    '''Eigenvalue self-consistent GW (evGW / evGW0) with frequency-sampled contour deformation.

    Attributes (in addition to GWFSCD):
        orbs : list of int
            Window of orbitals solved explicitly (default: all).  The others are scissor-shifted
            every cycle and enter G and W with the shifted energies.  The window must contain at
            least one occupied (virtual) orbital if any occupied (virtual) orbital lies outside.
        fs_polish : bool
            Default True here (see GWFSCD): each cycle's QP energies solve the canonical-CD
            equation for that cycle's G and W, so the iteration has no fit-noise floor.  Near
            self-consistency the residues already coincide with sample frequencies and
            polishing costs (almost) nothing.
        W0 : bool
            evGW0: keep the screening at the mean-field energies (default False, i.e. evGW).
        max_cycle : int
            Maximum number of evGW cycles (default 50).
        conv_tol : float
            Convergence threshold on max |e^(n+1) - e^(n)| over the window (Ha, default 1e-6).
        diis_space : int
            DIIS subspace size for the window energies (default 0 = plain iteration, which
            converges in a few cycles; DIIS can help for very large windows).
        diis_start_cycle : int
            First cycle (0-based) in which DIIS extrapolation is applied (default 1).
        diis_file : str or None

    Saved results:
        mo_energy : quasiparticle energies of all orbitals (window from GW, others
            scissor-shifted, frozen orbitals at mean-field values).
        converged, conv_inds, cycle, real_freqs (of the last cycle)
    '''

    def __init__(self, mf, frozen=None, auxbasis=None):
        super().__init__(mf, frozen=frozen, auxbasis=auxbasis)
        self.W0 = False
        self.fs_polish = True
        self.max_cycle = 50
        self.conv_tol = 1e-6
        self.diis_space = 0
        self.diis_start_cycle = 1
        self.diis_file = None
        self.cycle = None

    def dump_flags(self):
        super().dump_flags()
        log = logger.new_logger(self)
        log.info('evGW0 = %s', self.W0)
        log.info('max cycle = %d  conv_tol = %.1e', self.max_cycle, self.conv_tol)
        log.info('DIIS space = %d  start cycle = %d', self.diis_space, self.diis_start_cycle)

    def kernel(self):
        '''Run fsCD-evGW (or fsCD-evGW0 with W0 = True).'''
        self._check_input()
        cput0 = (time.process_time(), time.perf_counter())
        self.dump_flags()
        self.converged, self.conv_inds, self.mo_energy = kernel(self)
        logger.timer(self, 'EVGWFSCD', *cput0)
        return self.mo_energy
