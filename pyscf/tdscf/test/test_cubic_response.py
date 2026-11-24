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
from pyscf.tdscf.quadratic_response import QRTDDFT, gamma_parallel
from pyscf.tdscf.qr_base import _tpa3_strength


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


def _gamma_ff(xc, mf, k):
    '''Richardson-extrapolated d beta / dE_k of the analytic static beta: gamma[k].'''
    dm = mf.make_rdm1()

    def beta(E):
        qr = QRTDDFT(_run_mf(xc, E, dm))
        qr.verbose = 0
        qr.solver = 'direct'
        return qr.hyperpolarizability()

    e = np.eye(3)[k]
    ff = {h: (beta(h * e) - beta(-h * e)) / (2 * h) for h in (4e-3, 2e-3)}
    return (4 * ff[2e-3] - ff[4e-3]) / 3


def _rotate3(T, R):
    return np.einsum('ia,jb,kc,nabc->nijk', R, R, R, T)


class KnownValues(unittest.TestCase):
    def test_static_finite_field(self):
        # static cubic response = field derivative of the (already validated) static quadratic
        # response; for DFT this requires the fourth functional derivative h_xc
        for xc in ('HF', 'LDA', 'PBE'):
            mf = _run_mf(xc)
            qr = QRTDDFT(mf)
            qr.solver = 'direct'
            gamma = qr.second_hyperpolarizability()
            ref = _gamma_ff(xc, mf, 2)
            self.assertAlmostEqual(abs(gamma[2] - ref).max() / abs(ref).max(), 0, 5)
            # static gamma is fully symmetric (Kleinman)
            self.assertAlmostEqual(abs(gamma - gamma.transpose(1, 0, 2, 3)).max(), 0, 7)
            self.assertAlmostEqual(abs(gamma - gamma.transpose(0, 3, 2, 1)).max(), 0, 7)
            if xc != 'HF':
                qr.hyperkernel = False
                g_nok = qr.second_hyperpolarizability()
                self.assertGreater(abs(g_nok[2, 2, 2, 2] - gamma[2, 2, 2, 2]), 1.0)

    def test_permutation_symmetry(self):
        # <<A; B, C, D>>_{w1, w2, w3} is invariant under permutations of the
        # (operator, frequency) pairs (A, -ws), (B, w1), (C, w2), (D, w3)
        qr = QRTDDFT(_run_mf('PBE0'))
        qr.solver = 'direct'
        r = qr.get_dipole_mo()
        w1, w2, w3 = 0.03, 0.05, -0.02
        ws = w1 + w2 + w3
        c = qr.cubic_response(r, r, r, r, w1, w2, w3)
        self.assertAlmostEqual(abs(c - qr.cubic_response(r, r, r, r, -ws, w2, w3)
                                   .transpose(1, 0, 2, 3)).max(), 0, 8)
        self.assertAlmostEqual(abs(c - qr.cubic_response(r, r, r, r, w2, w1, w3)
                                   .transpose(0, 2, 1, 3)).max(), 0, 8)
        self.assertAlmostEqual(abs(c - qr.cubic_response(r, r, r, r, w1, w3, w2)
                                   .transpose(0, 1, 3, 2)).max(), 0, 8)
        self.assertAlmostEqual(abs(c - qr.cubic_response(r, r, r, r, w1, w2, -ws)
                                   .transpose(3, 1, 2, 0)).max(), 0, 8)

    def test_symmetry_reduction(self):
        # reusing identical perturbations (THG, DFWM) must not change the result
        qr = QRTDDFT(_run_mf('LDA'))
        qr.solver = 'direct'
        r = qr.get_dipole_mo()
        rc = [r.copy() for _ in range(4)]
        for w in ((0.04, 0.04, 0.04), (0.04, -0.04, 0.04)):
            self.assertAlmostEqual(abs(qr.cubic_response(r, r, r, r, *w)
                                       - qr.cubic_response(*rc, *w)).max(), 0, 9)

    def test_iterative_vs_direct(self):
        qr = QRTDDFT(_run_mf('PBE0'))
        qr.solver = 'direct'
        g_dir = qr.second_hyperpolarizability(0.02, 0.03, 0.04)
        qr.solver = 'iterative'
        g_it = qr.second_hyperpolarizability(0.02, 0.03, 0.04)
        self.assertAlmostEqual(abs(g_dir - g_it).max(), 0, 5)

    def test_dispersion(self):
        qr = QRTDDFT(_run_mf('PBE0'))
        g0 = gamma_parallel(qr.second_hyperpolarizability())
        g_thg = gamma_parallel(qr.second_hyperpolarizability(0.0428, 0.0428, 0.0428))
        g_dfwm = gamma_parallel(qr.second_hyperpolarizability(0.0428, -0.0428, 0.0428))
        self.assertGreater(g0, 0)
        self.assertGreater(g_dfwm, g0)   # dispersion increases gamma below resonance
        self.assertGreater(g_thg, g_dfwm)

    def test_tpa3_residue(self):
        mf = _run_mf('PBE0')
        qr = QRTDDFT(mf)
        qr.solver = 'direct'
        exci, T, delta = qr.three_photon_absorption(nstates=4)
        self.assertAlmostEqual(abs(qr.tpa3_frequencies.sum(axis=1) - exci).max(), 0, 12)
        # degenerate photons: fully symmetric tensor
        self.assertAlmostEqual(abs(T - T.transpose(0, 2, 1, 3)).max(), 0, 8)
        self.assertAlmostEqual(abs(T - T.transpose(0, 1, 3, 2)).max(), 0, 8)
        r = qr.get_dipole_mo()
        tdm = qr.transition_dipole()
        n = int(np.argmax(np.abs(tdm).max(axis=1) * np.abs(T).reshape(4, -1).max(axis=1)))
        ref = np.einsum('x,yzw->xyzw', tdm[n], T[n])
        errs = []
        for d in (1e-4, 1e-5):
            w = (exci[n] - d) / 3
            errs.append(abs(-d * qr.cubic_response(r, r, r, r, w, w, w) - ref).max())
        self.assertLess(errs[1], 2e-3 * abs(ref).max())
        self.assertAlmostEqual(errs[0] / errs[1], 10, 0)
        self.assertTrue(np.all(delta['linear'] >= 0))

    def test_tpa3_nondegenerate(self):
        mf = _run_mf('PBE0')
        qr = QRTDDFT(mf)
        qr.solver = 'direct'
        exci, T_deg, _ = qr.three_photon_absorption(nstates=3)
        _, T, _ = qr.three_photon_absorption(nstates=3, frequencies=(None, None, None))
        self.assertAlmostEqual(abs(T - T_deg).max(), 0, 9)

        w1 = 0.1
        _, T1, d1 = qr.three_photon_absorption(nstates=3, frequencies=(w1, None, None))
        f = qr.tpa3_frequencies
        self.assertAlmostEqual(abs(f.sum(axis=1) - exci).max(), 0, 12)
        self.assertAlmostEqual(abs(f[:, 1] - f[:, 2]).max(), 0, 12)
        _, T2, d2 = qr.three_photon_absorption(nstates=3, frequencies=(None, w1, None))
        self.assertAlmostEqual(abs(T1 - T2.transpose(0, 2, 1, 3)).max(), 0, 8)
        self.assertAlmostEqual(abs(d1['linear'] - d2['linear']).max(), 0, 10)
        self.assertGreater(abs(T1 - T_deg).max(), 1e-3)

        # residue with the first photon fixed
        r = qr.get_dipole_mo()
        tdm = qr.transition_dipole()
        n = int(np.argmax(np.abs(tdm).max(axis=1) * np.abs(T1).reshape(3, -1).max(axis=1)))
        ref = np.einsum('x,yzw->xyzw', tdm[n], T1[n])
        errs = []
        for d in (1e-4, 1e-5):
            w = (exci[n] - w1 - d) / 2
            errs.append(abs(-d * qr.cubic_response(r, r, r, r, w1, w, w) - ref).max())
        self.assertLess(errs[1], 2e-3 * abs(ref).max())
        self.assertAlmostEqual(errs[0] / errs[1], 10, 0)

        with self.assertRaises(ValueError):
            qr.three_photon_absorption(nstates=3, frequencies=(0.1, 0.1, 0.1))
        with self.assertRaises(ValueError):
            qr.three_photon_absorption(nstates=3, frequencies=(0.1, None))

    def test_tpa3_strength_average(self):
        rng = np.random.default_rng(3)
        # symmetric tensor: (3 F + 2 G) / 35
        T = rng.standard_normal((2, 3, 3, 3))
        S = sum(T.transpose((0,) + tuple(1 + np.array(p)))
                for p in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0))) / 6
        F = np.einsum('niij,nkkj->n', S, S)
        G = np.einsum('nijk,nijk->n', S, S)
        self.assertAlmostEqual(abs(_tpa3_strength(S) - (3 * F + 2 * G) / 35).max(), 0, 12)
        # rotational invariance (also for non-symmetric tensors)
        R = np.linalg.qr(rng.standard_normal((3, 3)))[0]
        self.assertAlmostEqual(abs(_tpa3_strength(_rotate3(T, R)) - _tpa3_strength(T)).max(), 0, 12)
        # orientational average by quadrature over the unit sphere (Lebedev-free check)
        x, wx = np.polynomial.legendre.leggauss(12)
        phi = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        e = np.array([[np.sqrt(1 - c * c) * np.cos(p), np.sqrt(1 - c * c) * np.sin(p), c]
                      for c in x for p in phi])
        w = np.repeat(wx, len(phi)) / (2 * len(phi))
        avg = np.einsum('g,ng->n', w, np.einsum('nijk,gi,gj,gk->ng', T, e, e, e)**2)
        self.assertAlmostEqual(abs(avg - _tpa3_strength(T)).max(), 0, 10)

    def test_tdhf_matches_bse(self):
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
        w = (0.02, 0.03, 0.04)
        self.assertAlmostEqual(abs(a.second_hyperpolarizability(*w)
                                   - b.second_hyperpolarizability(*w)).max(), 0, 8)


if __name__ == '__main__':
    print('Full tests for TDDFT cubic response')
    unittest.main()
