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
from pyscf.gw.gw_cd import GWCD, get_sigma_diag, get_WmnI_diag
from pyscf.gw import gw_fscd
from pyscf.gw.gw_fscd import GWFSCD


def setUpModule():
    global mol, mf, nocc
    mol = gto.M(atom=[[8, (0.0, 0.0, 0.0)], [1, (0.0, -0.7571, 0.5861)], [1, (0.0, 0.7571, 0.5861)]],
                basis='def2-svp', verbose=0)
    mf = dft.RKS(mol, xc='pbe').run()
    nocc = mol.nelectron // 2


def tearDownModule():
    global mol, mf
    del mol, mf


class KnownValues(unittest.TestCase):
    def test_aaa_rational(self):
        # AAA reproduces a rational function with poles near the real axis from samples on the
        # imaginary axis plus a few real points
        poles = np.array([0.4 - 1e-3j, 0.9 - 1e-3j, 2.5 - 1e-3j])
        res = np.array([0.3, -0.2, 0.5])
        f = lambda z: np.sum(res / (z[:, None] - poles), axis=1)
        z = np.concatenate([1j * np.linspace(0.01, 20, 40), np.array([0.0, 0.3, 1.2, 3.0])])
        fit = gw_fscd.aaa_fit(z, f(z))
        x = np.array([0.1, 0.55, 1.7, 2.2, 4.0])
        approx = np.array([gw_fscd.aaa_eval(*fit, xi) for xi in x])
        self.assertAlmostEqual(abs(approx - f(x)).max(), 0, 8)
        # exact at (and within 5e-6 of) a support point
        zj, fj, _ = fit
        self.assertEqual(gw_fscd.aaa_eval(*fit, zj[0] + 1e-7), fj[0])

    def test_real_grid(self):
        cand = [18.70, 18.72, 18.80, 19.18, 19.20, 0.02, 0.5, 0.53]
        grid = gw_fscd.build_real_grid(cand, t1=0.05, t2=1e-3)
        self.assertTrue(np.allclose(grid, [0.0, 0.5, 18.70, 18.80, 19.18]))
        # relative criterion dominates at large frequencies
        grid = gw_fscd.build_real_grid([100.0, 100.06, 100.2], t1=0.05, t2=1e-3)
        self.assertTrue(np.allclose(grid, [0.0, 100.0, 100.2]))
        new = gw_fscd._merge_new_points([0.51, 0.6, 18.74, 30.0], [0.0, 0.5, 18.7], 0.05, 1e-3)
        self.assertTrue(np.allclose(new, [0.6, 30.0]))

    def test_vs_canonical_cd(self):
        orbs = list(range(nocc + 3))
        cd = GWCD(mf)
        cd.orbs = orbs
        cd.kernel()
        gw = GWFSCD(mf)
        gw.orbs = orbs
        gw.kernel()
        self.assertTrue(gw.converged)
        # canonical CD reference values from pyscf/gw/test/test_gw_cd.py
        self.assertAlmostEqual(gw.mo_energy[nocc - 1], -0.41284735, 4)
        self.assertAlmostEqual(gw.mo_energy[nocc], 0.16574524, 4)
        self.assertAlmostEqual(gw.mo_energy[0], -19.53387986, 4)
        # core 1s and outer valence agree to about a meV (1e-4 Ha = 2.7 meV)
        for p in [0] + list(range(2, nocc + 3)):
            self.assertAlmostEqual(gw.mo_energy[p], cd.mo_energy[p], 4)
        # tighter real-axis grid converges to canonical CD.  The O 2s (orbital 1) is excluded:
        # its QP equation has several roots in the satellite region and Newton may land on a
        # different one; it is covered by the fs_exact_cutoff check below.
        gw = GWFSCD(mf)
        gw.orbs = orbs
        gw.fs_t1 = 0.005
        gw.kernel()
        well_defined = [0] + list(range(2, nocc + 3))
        self.assertAlmostEqual(abs(gw.mo_energy[well_defined] - cd.mo_energy[well_defined]).max(), 0, 5)
        # so does evaluating the small-frequency residues exactly at the default T1
        gw = GWFSCD(mf)
        gw.orbs = orbs
        gw.fs_exact_cutoff = 2.0
        gw.kernel()
        self.assertAlmostEqual(abs(gw.mo_energy[orbs] - cd.mo_energy[orbs]).max(), 0, 5)

    def test_sigma_c_vs_canonical_cd(self):
        # window around the 1s quasiparticle solution (-19.534 Ha)
        omegas = -19.534 + np.linspace(-0.05, 0.05, 6)
        gw = GWFSCD(mf)
        gw.orbs = [0]
        gw.fs_exact_cutoff = 2.0
        sigma = gw.get_sigma_c(0, omegas)
        Lpq = gw.Lpq
        Lia = np.ascontiguousarray(Lpq[:, :nocc, nocc:])
        qf, qw = gw_fscd._get_scaled_legendre_roots(gw.nw)
        WI = get_WmnI_diag(gw, [0], Lpq, Lia, qf, mf.mo_energy)
        ref = [get_sigma_diag(gw.ef, w, 0, mf.mo_energy, Lpq, Lia, WI[:, 0], qf, qw, gw.eta) for w in omegas]
        # only the self-residue (|w - e_0| ~ 0.8 a.u.) is exact; the valence residues at
        # ~19 a.u. come from the fits
        self.assertAlmostEqual(abs(sigma.real - np.real(ref)).max(), 0, 5)
        # the valence residues cluster: far fewer real sample frequencies than residues
        self.assertLess(gw._fs['samples'].freqs.size, len(omegas) * (nocc - 1))

    def test_frozen_and_noncontiguous_orbs(self):
        ref = GWFSCD(mf)
        ref.orbs = list(range(nocc + 2))
        ref.kernel()
        gw = GWFSCD(mf, frozen=1)
        gw.orbs = [2, nocc - 1, nocc + 1]
        gw.kernel()
        for p in gw.orbs:
            self.assertAlmostEqual(gw.mo_energy[p], ref.mo_energy[p], 3)


    def test_lowmem_ints(self):
        # only the ov block and the GW-orbital rows of Lpq, from an AO DF tensor on disk
        orbs = range(nocc - 3, nocc + 3)
        ref = GWFSCD(mf)
        ref.orbs = orbs
        ref.fs_exact_cutoff = 2.0
        ref.qpe_tol = 1e-12
        ref.kernel()
        gw = GWFSCD(mf)
        gw.orbs = orbs
        gw.fs_exact_cutoff = 2.0
        gw.qpe_tol = 1e-12
        gw.lowmem_ints = True
        gw.initialize_df()
        gw._cderi_tmpfile = None
        gw.with_df._cderi_to_save = gw_fscd.lib.NamedTemporaryFile().name   # force on disk
        gw.kernel()
        self.assertIsNone(gw.Lpq)
        win = gw._fs_ints['Lpq']
        self.assertIsInstance(win, gw_fscd._WindowLpq)
        L = ref.Lpq
        self.assertAlmostEqual(abs(win.ov - L[:, :nocc, nocc:]).max(), 0, 12)
        self.assertAlmostEqual(abs(win.rows - L[:, list(orbs)]).max(), 0, 12)
        e = np.asarray(list(orbs))
        # W and Sigma are identical to roundoff; the AAA fits of the sampled residues amplify
        # that roundoff to ~1e-8 Ha, so compare QP energies at 1e-6
        self.assertAlmostEqual(abs(gw.mo_energy[e] - ref.mo_energy[e]).max(), 0, 6)
        w = [0.3, 1.1]
        n = L.shape[1]
        Wa = gw_fscd.get_W_diag(L, nocc, mf.mo_energy, w, [(list(orbs), slice(0, n))], eta=ref.eta)[0]
        Wb = gw_fscd.get_W_diag(win, nocc, mf.mo_energy, w, [(list(orbs), slice(0, n))], eta=ref.eta)[0]
        self.assertAlmostEqual(abs(Wa - Wb).max(), 0, 12)

        # BSE quadratic response from the low-memory GW object builds its own blocks
        from pyscf.gw.bse_qr import QRBSE
        qa, qb = QRBSE(ref), QRBSE(gw)
        qa.solver = qb.solver = 'direct'
        self.assertAlmostEqual(abs(qa.Lpq - qb.Lpq).max(), 0, 12)
        ea = qa.get_excited_states(3)[0]
        eb = qb.get_excited_states(3)[0]
        self.assertAlmostEqual(abs(ea - eb).max(), 0, 5)

if __name__ == '__main__':
    print('Full tests for fsCD-GW')
    unittest.main()
