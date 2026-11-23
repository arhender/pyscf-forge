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
from pyscf import gto, scf, dft
from pyscf.tdscf import rhf as tdrhf
from pyscf.tdscf.quadratic_response import QRTDDFT


def setUpModule():
    global mol, ao_dip
    mol = gto.M(
        atom='O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692',
        basis='6-31g', verbose=0)
    ao_dip = mol.intor_symmetric('int1e_r', comp=3)


def tearDownModule():
    global mol, ao_dip
    del mol, ao_dip


def _run_mf(xc, E=None, dm0=None):
    mf = scf.RHF(mol) if xc == 'HF' else dft.RKS(mol, xc=xc)
    mf.conv_tol = 1e-14
    mf.conv_tol_grad = 1e-10
    mf.max_cycle = 200
    if E is not None:
        h = mf.get_hcore()
        mf.get_hcore = lambda *args: h + np.einsum('x,xuv->uv', E, ao_dip)
    mf.kernel(dm0=dm0)
    return mf


def _beta_ff(xc, mf, k):
    '''Richardson-extrapolated finite-field beta_zkk from KS dipole moments.'''
    dm = mf.make_rdm1()

    def mu_z(E):
        return -np.einsum('uv,vu->', ao_dip[2], _run_mf(xc, E, dm).make_rdm1())

    e = np.eye(3)[k]
    m0 = mu_z(np.zeros(3))
    ff = {h: (mu_z(h * e) - 2 * m0 + mu_z(-h * e)) / h**2 for h in (4e-3, 2e-3)}
    return (4 * ff[2e-3] - ff[4e-3]) / 3


class KnownValues(unittest.TestCase):
    def test_dense_matrices_vs_get_ab(self):
        for xc in ('HF', 'LDA', 'PBE0', 'CAMB3LYP'):
            mf = _run_mf(xc)
            qr = QRTDDFT(mf)
            apb, amb = qr.get_dense_apb_amb()
            a, b = tdrhf.get_ab(mf)
            dim = apb.shape[0]
            a = a.reshape(dim, dim)
            b = b.reshape(dim, dim)
            self.assertAlmostEqual(abs(apb - (a + b)).max(), 0, 9)
            self.assertAlmostEqual(abs(amb - (a - b)).max(), 0, 9)

    def test_static_finite_field(self):
        # static TDDFT quadratic response = second field derivative of the KS dipole,
        # which requires the XC hyperkernel
        for xc in ('LDA', 'PBE', 'PBE0'):
            mf = _run_mf(xc)
            qr = QRTDDFT(mf)
            beta = qr.hyperpolarizability()
            for k in (2, 1):
                self.assertAlmostEqual(beta[2, k, k], _beta_ff(xc, mf, k), 3)
            qr.hyperkernel = False
            beta_nok = qr.hyperpolarizability()
            self.assertGreater(abs(beta_nok[2, 1, 1] - beta[2, 1, 1]), 0.1)

    def test_iterative_vs_direct(self):
        qr = QRTDDFT(_run_mf('PBE0'))
        qr.solver = 'direct'
        b_dir = qr.hyperpolarizability(0.03, 0.05)
        qr.solver = 'iterative'
        b_it = qr.hyperpolarizability(0.03, 0.05)
        self.assertAlmostEqual(abs(b_dir - b_it).max(), 0, 6)

    def test_permutation_symmetry(self):
        # <<A; B, C>>_{w1, w2} = <<B; A, C>>_{-w1-w2, w2}, including the XC hyperkernel
        qr = QRTDDFT(_run_mf('PBE0'))
        r = qr.get_dipole_mo()
        q1 = qr.quadratic_response(r, r, r, 0.05, 0.08)
        q2 = qr.quadratic_response(r, r, r, -0.13, 0.08)
        self.assertAlmostEqual(abs(q1 - q2.transpose(1, 0, 2)).max(), 0, 9)

    def test_tpa_residue(self):
        mf = _run_mf('PBE0')
        qr = QRTDDFT(mf)
        qr.solver = 'direct'
        exci, T, delta = qr.two_photon_absorption(nstates=4)
        td = mf.TDDFT()
        td.nstates = 4
        td.conv_tol = 1e-10
        td.kernel()
        self.assertAlmostEqual(abs(exci - td.e).max(), 0, 7)
        r = qr.get_dipole_mo()
        tdm = qr.transition_dipole()
        n = int(np.argmax(np.abs(tdm).max(axis=1) * np.abs(T).reshape(4, -1).max(axis=1)))
        ref = np.einsum('x,yz->xyz', tdm[n], T[n])
        errs = [abs(-d * qr.quadratic_response(r, r, r, (exci[n] - d) / 2, (exci[n] - d) / 2) - ref).max()
                for d in (1e-4, 1e-5)]
        self.assertLess(errs[1], 2e-3 * abs(ref).max())
        self.assertAlmostEqual(errs[0] / errs[1], 10, 0)
        self.assertTrue(np.all(delta['linear'] >= 0))

    def test_tdhf_matches_bse_qr(self):
        try:
            from pyscf.gw.bse_qr import QRBSE
        except ImportError:
            self.skipTest('pyscf.gw.bse_qr not available')
        mf = scf.RHF(mol).density_fit()
        mf.conv_tol = 1e-12
        mf.kernel()
        a = QRTDDFT(mf)
        b = QRBSE(mf, screening=False)
        a.solver = b.solver = 'direct'
        self.assertAlmostEqual(abs(a.hyperpolarizability(0.03, 0.05) - b.hyperpolarizability(0.03, 0.05)).max(), 0, 9)


if __name__ == '__main__':
    print('Full tests for TDDFT quadratic response')
    unittest.main()
