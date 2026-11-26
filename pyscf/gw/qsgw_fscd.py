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
Spin-restricted quasiparticle self-consistent GW (qsGW) without analytic continuation:
off-diagonal elements on the imaginary axis, diagonal elements by frequency-sampled
contour deformation (fsCD).

qsGW (pyscf.gw.qsgw, Kotani, van Schilfgaarde, Faleev, PRB 76, 165106 (2007)) replaces the
dynamical self-energy by a static Hermitian matrix in the basis of the current orbitals.  In
mode 'b' (the pyscf default)

    Sigma~_pp = Re Sigma_pp(e_p),      Sigma~_pq = Re [Sigma_pq(E_F)]_herm  (p != q),

and in mode 'c' all elements are taken at E_F.  pyscf.gw.qsgw obtains both from an analytic
continuation (Pade or two-pole) of the imaginary-axis self-energy, which is ill-conditioned:
rounding-level changes of Sigma(iw) can move core and high-virtual QP energies by up to mHa,
and the two-pole fits of these states often do not converge.  Here

  * Sigma_pq(E_F + i eta) is evaluated directly with the imaginary-axis quadrature of the
    correlation self-energy (E_F + i eta is itself a point on the imaginary axis), i.e. the
    same quantity the continuation of pyscf.gw.qsgw approximates, without continuation;
  * Sigma_pp(e_p) is evaluated on the real axis by contour deformation, with the residues
    from frequency-sampled W (pyscf.gw.gw_fscd: AAA fits on a sparse real-frequency grid,
    residues below fs_exact_cutoff exactly), rebuilt every cycle for the current orbitals
    and orbital energies.  The pole of G at omega = e_p itself is treated as in fsCD-evGW
    (half residue plus the analytic W(i0) term of the imaginary-axis integral).

Everything else (Hamiltonian, frozen orbitals, windows and the optional scissor, DIIS,
convergence) is that of pyscf.gw.qsgw_lowmem.QSGWLowMem.

Multiple solutions: Sigma_pp(e_p) at the orbital's own energy makes the diagonal condition a
QP-like equation per orbital.  For the highest virtuals of small basis sets (H2O/def2-SVP:
the seven states above ~60 eV) it has several solutions, already with canonical CD: the
converged energies depend on the starting point and on meV-level differences of Sigma (fsCD
vs exact residues pick different ones, eV apart), as for the QP equations of evGW.  Valence
and core states are not affected (fsCD vs canonical CD: ~0.3 meV valence, ~1 meV core).

Not implemented: mode 'a' (needs off-diagonal residues Sigma_pq(e_p), i.e. off-diagonal W
at the residue frequencies), smeared (fractional-occupation) references, and out-of-core
integrals (the fsCD part keeps the MO integrals in memory).
'''

import numpy as np

from pyscf.lib import logger
from pyscf.gw import gw_fscd
from pyscf.gw.gw_ac_lowmem import get_sigma_lowmem
from pyscf.gw.qsgw_lowmem import QSGWLowMem


def get_static_sigma_fscd(gw, Lpq, orbs_frz, mo_energy_frz, mo_coeff_frz, mo_occ_frz, ef,
                          quad_freqs, quad_wts):
    '''Static Hermitian qsGW self-energy (len(orbs), len(orbs)) of the current orbitals,
    mode 'b' or 'c', without analytic continuation (see module docstring).'''
    log = logger.new_logger(gw)
    # Sigma_pq(E_F + i eta): one point of the imaginary-axis self-energy
    s = get_sigma_lowmem(gw, orbs_frz, quad_freqs, quad_wts, ef, mo_energy_frz, mo_coeff_frz,
                         mo_occ=mo_occ_frz, iw_cutoff=None, eval_freqs=np.array([gw.eta]),
                         fullsigma=True, Lpq=Lpq)[0][:, :, 0]
    nfrz_sigma = 0.5 * (s + s.T.conj())
    if gw.mode == 'c':
        return nfrz_sigma

    # mode 'b': diagonal Sigma_pp(e_p) by fsCD with the current orbitals in G and W
    if Lpq is None:
        raise NotImplementedError('fsCD-qsGW needs the MO integrals in memory (outcore not supported)')
    gw._fs = None
    gw._fs_ints = {'orbs_frz': list(orbs_frz), 'Lpq': Lpq, 'mo_energy_mf': mo_energy_frz}
    gw_fscd.setup(gw, mo_energy_g=mo_energy_frz, mo_energy_w=mo_energy_frz, ef=ef)
    for k, p in enumerate(orbs_frz):
        nfrz_sigma[k, k] = gw_fscd.get_sigma_c(gw, k, mo_energy_frz[p]).real
    log.debug('fsCD-qsGW: %d real sample frequencies', gw._fs['samples'].freqs.size)
    gw.real_freqs = gw._fs['samples'].freqs
    gw._fs = gw._fs_ints = None
    return nfrz_sigma


class QSGWFSCD(QSGWLowMem):
    '''qsGW (mode 'b' or 'c') without analytic continuation; diagonal elements by fsCD.

    Same attributes as pyscf.gw.qsgw_lowmem.QSGWLowMem (ac, ac_* unused), plus the fsCD
    controls of pyscf.gw.gw_fscd.GWFSCD:
        fs_t1, fs_t2 : real-axis grid merge thresholds (default 0.05 a.u., 1e-3)
        fs_aaa_rtol, fs_aaa_max_terms : AAA fit controls (default 1e-6, 100).  With the
            tolerance of GWFSCD (1e-13) the greedy AAA support selection depends on rounding,
            and self-consistency amplifies this: high virtuals of H2O/def2-SVP varied by up
            to 4 eV between identical runs; with 1e-6 runs agree exactly and states below
            ~55 eV agree with all-exact residues (canonical CD) to ~2 meV.
        fs_exact_cutoff : residues with |w - e_m| below this (a.u.) are evaluated exactly
            (default 2.0, as recommended for fsCD-evGW; None for pure frequency sampling)
    eta (default 5e-3 as in pyscf.gw.qsgw) is the broadening of both the off-diagonal
    evaluation point E_F + i eta and the real-axis response function.

    Saved results (in addition to QSGWLowMem): real_freqs, the real-axis sample frequencies
    of the last cycle.
    '''

    def __init__(self, mf, frozen=None, auxbasis=None):
        QSGWLowMem.__init__(self, mf, frozen=frozen, auxbasis=auxbasis)
        self.fs_t1 = 0.05
        self.fs_t2 = 1e-3
        self.fs_aaa_rtol = 1e-6       # (1e-13 of GWFSCD: not reproducible here, see docstring)
        self.fs_aaa_max_terms = 100
        self.fs_exact_cutoff = 2.0
        self.real_freqs = None
        self._fs = None
        self._fs_ints = None

    def dump_flags(self):
        QSGWLowMem.dump_flags(self)
        log = logger.new_logger(self)
        log.info('fsCD-qsGW: off-diagonal Sigma at E_F + i eta on the imaginary axis, diagonal by fsCD')
        log.info('fsCD real-grid thresholds T1 = %.3g a.u., T2 = %.3g', self.fs_t1, self.fs_t2)
        log.info('fsCD AAA rtol = %.1e, max terms = %d', self.fs_aaa_rtol, self.fs_aaa_max_terms)
        log.info('fsCD exact residues below %s a.u.', self.fs_exact_cutoff)

    def get_static_sigma(self, Lpq, orbs_frz, mo_energy_frz, mo_coeff_frz, mo_occ_frz, ef,
                         quad_freqs, quad_wts, eval_freqs):
        return get_static_sigma_fscd(self, Lpq, orbs_frz, mo_energy_frz, mo_coeff_frz, mo_occ_frz, ef,
                                     quad_freqs, quad_wts)

    def kernel(self):
        '''Run a quasiparticle self-consistent GW calculation (mode 'b' or 'c').'''
        if self.mode not in ('b', 'c'):
            raise NotImplementedError("fsCD-qsGW implements mode 'b' and 'c'")
        if hasattr(self._scf, 'sigma'):
            raise NotImplementedError('fsCD-qsGW does not support smeared references')
        return QSGWLowMem.kernel(self)
