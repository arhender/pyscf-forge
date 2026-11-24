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
from pyscf.gw.gw_ac import GWAC
from pyscf.gw.bse import BSE
from pyscf.gw.bse_qr import QRBSE, beta_parallel
from pyscf.gw.qp_shift import scissor_shift


def setUpModule():
    global mol, mf_hf, mf_ks, gw
    mol = gto.M(
        atom='O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692',
        basis='6-31g', verbose=0)
    mf_hf = scf.RHF(mol).density_fit()
    mf_hf.conv_tol = 1e-12
    mf_hf.kernel()
    mf_ks = dft.RKS(mol, xc='pbe0')
    mf_ks.conv_tol = 1e-12
    mf_ks.kernel()
    gw = GWAC(mf_ks)
    gw.kernel()


def tearDownModule():
    global mol, mf_hf, mf_ks, gw
    del mol, mf_hf, mf_ks, gw


def _finite_field_dipole(mf, E):
    ao = mf.mol.intor_symmetric('int1e_r', comp=3)
    hcore = mf.get_hcore()
    m = scf.RHF(mf.mol).density_fit()
    m.conv_tol = 1e-13
    m.get_hcore = lambda *args: hcore + np.einsum('x,xuv->uv', E, ao)
    m.kernel(dm0=mf.make_rdm1())
    return -np.einsum('xuv,vu->x', ao, m.make_rdm1())


class KnownValues(unittest.TestCase):
    def test_kernel_matches_bse_contraction(self):
        for screening in (False, True):
            qr = QRBSE(mf_hf, screening=screening).build()
            o, v = qr.nocc, qr.nmo - qr.nocc
            apb, amb = qr.get_dense_apb_amb()
            rng = np.random.default_rng(7)
            zp, zm = rng.random(o * v), rng.random(o * v)
            D = qr._ovvo_to_full(((zp + zm) / 2).reshape(1, o, v), ((zp - zm) / 2).reshape(1, o, v))[0]
            G = qr.get_kernel(D)
            gai, gia = G[o:, :o].T.ravel(), G[:o, o:].ravel()
            d = qr._orb_diff()
            self.assertAlmostEqual(abs(apb @ zp - d * zp - (gai + gia)).max(), 0, 10)
            self.assertAlmostEqual(abs(amb @ zm - d * zm - (gai - gia)).max(), 0, 10)

    def test_iterative_vs_direct(self):
        qr = QRBSE(gw)
        qr.solver = 'direct'
        a_dir = qr.polarizability(0.1)
        b_dir = qr.hyperpolarizability(0.03, 0.05)
        qr.solver = 'iterative'
        a_it = qr.polarizability(0.1)
        b_it = qr.hyperpolarizability(0.03, 0.05)
        self.assertAlmostEqual(abs(a_dir - a_it).max(), 0, 7)
        self.assertAlmostEqual(abs(b_dir - b_it).max(), 0, 6)

    def test_tdhf_limit_finite_field(self):
        # unscreened kernel + HF energies = DF-TDHF, whose quadratic response is the
        # second field-derivative of the DF-RHF dipole moment
        qr = QRBSE(mf_hf, screening=False)
        alpha = qr.polarizability()
        beta = qr.hyperpolarizability()
        h = 2e-3
        z = np.array([0, 0, 1.0])
        y = np.array([0, 1.0, 0])
        mu0 = _finite_field_dipole(mf_hf, np.zeros(3))
        mpz, mmz = _finite_field_dipole(mf_hf, h * z), _finite_field_dipole(mf_hf, -h * z)
        mpy, mmy = _finite_field_dipole(mf_hf, h * y), _finite_field_dipole(mf_hf, -h * y)
        self.assertAlmostEqual(alpha[2, 2], (mpz[2] - mmz[2]) / (2 * h), 3)
        self.assertAlmostEqual(beta[2, 2, 2] / ((mpz[2] - 2 * mu0[2] + mmz[2]) / h**2), 1, 3)
        self.assertAlmostEqual(beta[2, 1, 1] / ((mpy[2] - 2 * mu0[2] + mmy[2]) / h**2), 1, 3)
        # static TDHF beta is fully symmetric (Kleinman)
        self.assertAlmostEqual(beta[1, 1, 2], beta[2, 1, 1], 8)

    def test_explicit_second_order_density(self):
        qr = QRBSE(gw)
        r = qr.get_dipole_mo()
        rb = qr.first_order(r, 0.05)
        rc = qr.first_order(r, 0.08)
        q = qr.quadratic_response(r, r, r, 0.05, 0.08)
        D2 = qr.second_order_density(rb, rc, 1, 1)
        self.assertAlmostEqual(2 * np.einsum('pq,pq', r[2], D2), q[2, 1, 1], 8)
        D2 = qr.second_order_density(rb, rc, 1, 2)
        self.assertAlmostEqual(2 * np.einsum('pq,pq', r[1], D2), q[1, 1, 2], 8)

    def test_permutation_symmetry(self):
        # <<A; B, C>>_{w1, w2} = <<B; A, C>>_{-w1-w2, w2}
        qr = QRBSE(gw)
        r = qr.get_dipole_mo()
        q1 = qr.quadratic_response(r, r, r, 0.05, 0.08)
        q2 = qr.quadratic_response(r, r, r, -0.13, 0.08)
        self.assertAlmostEqual(abs(q1 - q2.transpose(1, 0, 2)).max(), 0, 8)

    def test_tpa_residue(self):
        bse = BSE(gw)
        bse.nroot = 4
        e_ref = bse.kernel('s')[0]
        qr = QRBSE(bse)
        qr.solver = 'direct'
        exci, T, delta = qr.two_photon_absorption(nstates=4)
        self.assertAlmostEqual(abs(exci - e_ref[:4]).max(), 0, 7)
        r = qr.get_dipole_mo()
        tdm = qr.transition_dipole()
        n = int(np.argmax(np.abs(tdm).max(axis=1) * np.abs(T).reshape(4, -1).max(axis=1)))
        ref = np.einsum('x,yz->xyz', tdm[n], T[n])
        errs = []
        for d in (1e-4, 1e-5):
            w = (exci[n] - d) / 2
            errs.append(abs(-d * qr.quadratic_response(r, r, r, w, w) - ref).max())
        self.assertLess(errs[1], 2e-3 * abs(ref).max())
        self.assertAlmostEqual(errs[0] / errs[1], 10, 0)
        self.assertTrue(np.all(delta['linear'] >= 0))

    def test_beta_parallel_regression(self):
        qr = QRBSE(gw)
        mu = mf_ks.dip_moment(verbose=0)
        b = beta_parallel(qr.hyperpolarizability(), mu)
        b_shg = beta_parallel(qr.hyperpolarizability(0.0428, 0.0428), mu)
        self.assertLess(b, 0)
        self.assertLess(b_shg, b)  # dispersion increases |beta| below resonance


    def test_scissor_shift(self):
        nocc = mol.nelectron // 2
        win = list(range(nocc - 2, nocc + 2))
        gww = GWAC(mf_ks)
        gww.orbs = win
        gww.kernel()
        # G0W0 corrections are orbital-wise, so the window reproduces the full run
        self.assertAlmostEqual(abs(gww.mo_energy[win] - gw.mo_energy[win]).max(), 0, 6)
        e = scissor_shift(gww)
        corr = gww.mo_energy - mf_ks.mo_energy
        self.assertAlmostEqual(abs(e[win] - gww.mo_energy[win]).max(), 0, 12)
        below = np.arange(nocc - 2)
        above = np.arange(nocc + 2, mf_ks.mo_energy.size)
        self.assertAlmostEqual(abs(e[below] - mf_ks.mo_energy[below] - corr[nocc - 2]).max(), 0, 12)
        self.assertAlmostEqual(abs(e[above] - mf_ks.mo_energy[above] - corr[nocc + 1]).max(), 0, 12)
        # full window: unchanged
        self.assertAlmostEqual(abs(scissor_shift(gw) - gw.mo_energy).max(), 0, 12)
        # GW object input applies the shift without touching gw; raw BSE input is rejected
        qr = QRBSE(gww)
        self.assertAlmostEqual(abs(qr.mo_energy - e).max(), 0, 12)
        self.assertEqual(np.count_nonzero(gww.mo_energy == 0.0), mf_ks.mo_energy.size - len(win))
        self.assertRaises(RuntimeError, QRBSE, BSE(gww))
        # low-lying physics is insensitive to the shifted orbitals
        mu = mf_ks.dip_moment(verbose=0)
        b_full = beta_parallel(QRBSE(gw).hyperpolarizability(), mu)
        b_win = beta_parallel(qr.hyperpolarizability(), mu)
        self.assertAlmostEqual(b_win / b_full, 1, 1)
        qr.solver = 'direct'
        qf = QRBSE(gw)
        qf.solver = 'direct'
        # shifting most orbitals rigidly moves the lowest excitations by ~10 meV here
        self.assertLess(abs(qr.get_excited_states(3)[0] - qf.get_excited_states(3)[0]).max(), 2e-3)


    def test_screened_finite_field(self):
        # static-screened BSE response with neglected hyperkernel is the response of the
        # time-independent fixed-W model F = diag(eps_QP) + G_W[D - D0] + E.r; check static
        # alpha and beta against finite-field derivatives of that model's dipole moment
        qr = QRBSE(gw)
        qr.solver = 'direct'
        o, n = qr.nocc, qr.nmo
        r = qr.get_dipole_mo()
        D0 = np.zeros((n, n))
        D0[:o, :o] = np.eye(o)

        def mu(E):
            D = D0.copy()
            for _ in range(500):
                F = np.diag(qr.mo_energy) + qr.get_kernel(D - D0) + np.einsum('x,xpq->pq', E, r)
                C = np.linalg.eigh(F)[1]
                Dn = C[:, :o] @ C[:, :o].T
                if abs(Dn - D).max() < 1e-13:
                    break
                D = Dn
            return -2 * np.einsum('xpq,pq->x', r, D)

        alpha = qr.polarizability()
        beta = qr.hyperpolarizability()
        h = 2e-3
        z = np.array([0, 0, 1.0])
        y = np.array([0, 1.0, 0])
        m0, mpz, mmz, mpy, mmy = mu(np.zeros(3)), mu(h * z), mu(-h * z), mu(h * y), mu(-h * y)
        self.assertAlmostEqual(alpha[2, 2], (mpz[2] - mmz[2]) / (2 * h), 3)
        self.assertAlmostEqual(beta[2, 2, 2] / ((mpz[2] - 2 * m0[2] + mmz[2]) / h**2), 1, 3)
        self.assertAlmostEqual(beta[2, 1, 1] / ((mpy[2] - 2 * m0[2] + mmy[2]) / h**2), 1, 3)

    def test_cubic_screened_finite_field(self):
        # static BSE cubic response = third field derivative of the fixed-W model dipole
        qr = QRBSE(gw)
        qr.solver = 'direct'
        o, n = qr.nocc, qr.nmo
        r = qr.get_dipole_mo()
        D0 = np.zeros((n, n))
        D0[:o, :o] = np.eye(o)

        def mu(E):
            D = D0.copy()
            for _ in range(500):
                F = np.diag(qr.mo_energy) + qr.get_kernel(D - D0) + np.einsum('x,xpq->pq', E, r)
                C = np.linalg.eigh(F)[1]
                Dn = C[:, :o] @ C[:, :o].T
                if abs(Dn - D).max() < 1e-13:
                    break
                D = Dn
            return -2 * np.einsum('xpq,pq->x', r, D)

        def d3(k, h):
            e = np.eye(3)[k] * h
            return (mu(2 * e) - 2 * mu(e) + 2 * mu(-e) - mu(-2 * e)) / (2 * h**3)

        gamma = qr.second_hyperpolarizability()
        for k in (2, 1):
            # Richardson error is O(h^4): ~1e-3 relative at (2e-2, 1e-2), ~6e-6 at (1e-2, 5e-3)
            ref = (4 * d3(k, 5e-3) - d3(k, 1e-2)) / 3
            self.assertAlmostEqual(gamma[k, k, k, k] / ref[k], 1, 4)
            self.assertAlmostEqual(gamma[3 - k, k, k, k], ref[3 - k], 3)

    def test_cubic_permutation_symmetry(self):
        # <<A; B, C, D>>_{w1, w2, w3} = <<B; A, C, D>>_{-ws, w2, w3}
        qr = QRBSE(gw)
        r = qr.get_dipole_mo()
        c1 = qr.cubic_response(r, r, r, r, 0.03, 0.05, -0.02)
        c2 = qr.cubic_response(r, r, r, r, -0.06, 0.05, -0.02)
        self.assertAlmostEqual(abs(c1 - c2.transpose(1, 0, 2, 3)).max(), 0, 7)

    def test_tpa3_residue(self):
        qr = QRBSE(gw)
        qr.solver = 'direct'
        exci, T, delta = qr.three_photon_absorption(nstates=3)
        r = qr.get_dipole_mo()
        tdm = qr.transition_dipole()
        n = int(np.argmax(np.abs(tdm).max(axis=1) * np.abs(T).reshape(3, -1).max(axis=1)))
        ref = np.einsum('x,yzw->xyzw', tdm[n], T[n])
        errs = []
        for d in (1e-4, 1e-5):
            w = (exci[n] - d) / 3
            errs.append(abs(-d * qr.cubic_response(r, r, r, r, w, w, w) - ref).max())
        self.assertLess(errs[1], 2e-3 * abs(ref).max())
        self.assertAlmostEqual(errs[0] / errs[1], 10, 0)
        self.assertTrue(np.all(delta['linear'] >= 0))


if __name__ == '__main__':
    print('Full tests for BSE quadratic response')
    unittest.main()
