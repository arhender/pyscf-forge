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

import unittest
import numpy as np
from pyscf import gto, dft
from pyscf.gw import gw_fscd
from pyscf.gw.evgw_fscd import EVGWFSCD


def setUpModule():
    global mol, mf, nocc, nmo, window, ev
    mol = gto.M(atom=[[8, (0.0, 0.0, 0.0)], [1, (0.0, -0.7571, 0.5861)], [1, (0.0, 0.7571, 0.5861)]],
                basis='def2-svp', verbose=0)
    mf = dft.RKS(mol, xc='pbe0').run()
    nocc = mol.nelectron // 2
    nmo = mf.mo_energy.size
    window = list(range(nocc - 2, nocc + 3))
    ev = _run(orbs=window)


def tearDownModule():
    global mol, mf, ev
    del mol, mf, ev


def _run(**kw):
    gw = EVGWFSCD(mf)
    gw.fs_exact_cutoff = 2.0
    gw.conv_tol = 1e-6
    for k, v in kw.items():
        setattr(gw, k, v)
    gw.kernel()
    return gw


def _fixed_point_error(gw, e_g, e_w):
    '''Max |e_p - (QP solution with Sigma built from e_g in G and e_w in W)| over gw.orbs.'''
    setup = gw_fscd.setup(gw, mo_energy_g=e_g, mo_energy_w=e_w, ef=gw.get_ef(mo_energy=e_g))
    assert setup is gw._fs
    _, _, qp = gw_fscd.solve_qp(gw, guess=gw.mo_energy)
    return abs(qp[gw.orbs] - gw.mo_energy[gw.orbs]).max()


class KnownValues(unittest.TestCase):
    def test_window_converged_fixed_point(self):
        self.assertTrue(ev.converged)
        e = ev.mo_energy.copy()
        # self-consistent with the complete, scissor-shifted spectrum in G and W
        self.assertLess(_fixed_point_error(ev, e, e), 1e-6)
        # ... and not with the orbitals outside the window left at their mean-field energies
        # in the screening, i.e. the shifted orbitals do enter W
        e_w = e.copy()
        outside = np.setdiff1d(np.arange(nmo), window)
        e_w[outside] = mf.mo_energy[outside]
        self.assertGreater(_fixed_point_error(ev, e, e_w), 1e-4)
        ev.mo_energy = e

    def test_scissor_shift(self):
        e = ev.mo_energy
        d = e - mf.mo_energy
        lo, hi = window[0], window[-1]
        # occupied below the window: correction of the lowest window orbital
        self.assertTrue(np.allclose(d[:lo], d[lo]))
        # virtuals above the window: correction of the highest window orbital
        self.assertTrue(np.allclose(d[hi + 1:], d[hi]))
        # the shift is substantial (the test is meaningful)
        self.assertGreater(abs(d[0]), 1e-2)
        self.assertGreater(abs(d[-1]), 1e-2)

    def test_core_plus_valence_window(self):
        # non-contiguous window: the 2s orbital must take the valence (HOMO-2) correction,
        # not the 1s core correction
        gw = _run(orbs=[0] + window)
        self.assertTrue(gw.converged)
        d = gw.mo_energy - mf.mo_energy
        self.assertAlmostEqual(d[1], d[window[0]], 10)
        self.assertGreater(abs(d[0] - d[1]), 0.1)
        # adding the 1s to the window changes the valence window only slightly
        self.assertAlmostEqual(abs(gw.mo_energy[window] - ev.mo_energy[window]).max(), 0, 3)

    def test_evgw0(self):
        gw = _run(orbs=window, W0=True)
        self.assertTrue(gw.converged)
        e = gw.mo_energy.copy()
        self.assertLess(_fixed_point_error(gw, e, mf.mo_energy), 1e-6)
        # evGW0 opens the gap less than evGW
        gap0 = e[nocc] - e[nocc - 1]
        gap = ev.mo_energy[nocc] - ev.mo_energy[nocc - 1]
        self.assertLess(gap0, gap)
        self.assertGreater(gap0, mf.mo_energy[nocc] - mf.mo_energy[nocc - 1])

    def test_vs_canonical_cd(self):
        # every residue evaluated exactly = canonical-CD evGW
        ref = _run(orbs=window, fs_exact_cutoff=1e9)
        self.assertAlmostEqual(abs(ref.mo_energy - ev.mo_energy).max(), 0, 6)

    def test_unshifted_iteration_vs_ac_evgw(self):
        # pyscf's AC-based EVGW leaves the orbitals outside the window at their mean-field
        # energies; iterating the fsCD building blocks the same way must reproduce it
        from pyscf.gw.evgw import EVGW
        ac = EVGW(mf)
        ac.orbs = window
        ac.kernel()
        gw = EVGWFSCD(mf)
        gw.orbs = window
        gw.fs_exact_cutoff = 2.0
        gw._check_input()
        e = mf.mo_energy.copy()
        for _ in range(12):
            gw_fscd.setup(gw, e, e, gw.get_ef(mo_energy=e))
            _, _, qp = gw_fscd.solve_qp(gw, guess=e)
            e[window] = qp[window]
        # the Pade reference itself varies by up to ~5e-6 Ha between identical runs (rounding
        # from threaded reductions, amplified by the Pade continuation), so 1e-5 was flaky
        self.assertAlmostEqual(abs(e[window] - ac.mo_energy[window]).max(), 0, 4)
        # the scissor shift changes the result: it is not a no-op
        self.assertGreater(abs(ev.mo_energy[window] - e[window]).max(), 1e-3)

    def test_diis(self):
        gw = _run(orbs=window, diis_space=4)
        self.assertTrue(gw.converged)
        self.assertAlmostEqual(abs(gw.mo_energy - ev.mo_energy).max(), 0, 5)

    def test_frozen_core(self):
        gw = _run(orbs=window, frozen=1)
        self.assertTrue(gw.converged)
        self.assertEqual(gw.mo_energy[0], mf.mo_energy[0])
        # frozen 1s barely affects the valence window
        self.assertAlmostEqual(abs(gw.mo_energy[window] - ev.mo_energy[window]).max(), 0, 3)


if __name__ == '__main__':
    print('Full tests for fsCD-evGW')
    unittest.main()
