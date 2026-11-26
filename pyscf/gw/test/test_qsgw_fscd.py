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

'''qsGW with off-diagonal Sigma on the imaginary axis and diagonal Sigma by fsCD (QSGWFSCD).'''

import unittest
from pyscf import gto, dft, lib, scf
from pyscf.gw.qsgw_lowmem import QSGWLowMem
from pyscf.gw.qsgw_fscd import QSGWFSCD


def setUpModule():
    global mol, mf, nocc
    mol = gto.M(atom='O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692', basis='6-31g', verbose=0)
    mf = dft.RKS(mol, xc='pbe0').density_fit()
    mf.conv_tol = 1e-12
    mf.kernel()
    nocc = mol.nelectron // 2


def tearDownModule():
    global mol, mf
    del mol, mf


def _run(cls, **kw):
    g = cls(mf)
    g.verbose = 0
    for k, v in kw.items():
        setattr(g, k, v)
    with lib.with_omp_threads(1):
        g.kernel()
    return g


class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fs = _run(QSGWFSCD)

    def test_converged(self):
        self.assertTrue(self.fs.converged)

    def test_vs_exact_residues(self):
        # all residues exact = canonical contour deformation for the diagonal
        ex = _run(QSGWFSCD, fs_exact_cutoff=1e9)
        self.assertAlmostEqual(abs(self.fs.mo_energy[:11] - ex.mo_energy[:11]).max(), 0, 4)

    def test_valence_vs_analytic_continuation(self):
        # pyscf qsGW (mode b) continues Sigma(iw) analytically; near the gap the continuation
        # is accurate to about a meV (H2O: 1-2 meV for 6-31G and def2-SVP), so this is a
        # physical agreement check (5 meV), not an exactness test
        ac = _run(QSGWLowMem, ac='pade')
        val = list(range(nocc - 3, nocc + 2))
        self.assertLess(abs(self.fs.mo_energy[val] - ac.mo_energy[val]).max(), 5e-3 / 27.211386)

    def test_mode_c(self):
        # mode c only needs Sigma(E_F + i eta): continuation and direct evaluation agree
        a = _run(QSGWFSCD, mode='c')
        b = _run(QSGWLowMem, mode='c', ac='pade')
        self.assertAlmostEqual(abs(a.mo_energy - b.mo_energy).max(), 0, 7)

    def test_static_sigma_structure(self):
        S = self.fs.sigma_static
        self.assertAlmostEqual(abs(S - S.T).max(), 0, 12)

    def test_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            _run(QSGWFSCD, mode='a')
        smf = scf.addons.smearing_(dft.RKS(mol, xc='pbe').density_fit(), sigma=0.01, method='fermi')
        smf.kernel()
        g = QSGWFSCD(smf)
        g.verbose = 0
        with self.assertRaises(NotImplementedError):
            g.kernel()


if __name__ == '__main__':
    print('Full tests for fsCD-qsGW')
    unittest.main()
