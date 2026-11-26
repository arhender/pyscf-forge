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

'''Low-memory GW-AC, GW-CD, evGW, qsGW and BSE (pyscf.gw.*_lowmem) against upstream.'''

import unittest
import numpy as np
from pyscf import gto, dft, lib, scf
from pyscf.gw import gw_ac, gw_ac_lowmem
from pyscf.gw.gw_ac import GWAC, set_frozen_orbs, _mo_energy_without_core, _mo_without_core
from pyscf.gw.gw_ac_lowmem import GWACLowMem, get_sigma_lowmem
from pyscf.gw.gw_cd import GWCD
from pyscf.gw.gw_cd_lowmem import GWCDLowMem
from pyscf.gw.gw_fscd import GWFSCD
from pyscf.gw.evgw import EVGW
from pyscf.gw.evgw_lowmem import EVGWLowMem
from pyscf.gw.qsgw import QSGW
from pyscf.gw.qsgw_lowmem import QSGWLowMem
from pyscf.gw.bse import BSE
from pyscf.gw.bse_lowmem import BSELowMem
from pyscf.gw.utils.ac_grid import _get_scaled_legendre_roots


def setUpModule():
    global mol, mf, mf_small
    mol = gto.M(atom='O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692', basis='def2-svp', verbose=0)
    mf = dft.RKS(mol, xc='pbe0').density_fit()
    mf.conv_tol = 1e-12
    mf.kernel()
    # qsGW fits every element of Sigma(iw) (two-pole least squares): smaller basis
    mol_small = gto.M(atom='O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692', basis='6-31g', verbose=0)
    mf_small = dft.RKS(mol_small, xc='pbe0').density_fit()
    mf_small.conv_tol = 1e-12
    mf_small.kernel()


def tearDownModule():
    global mol, mf, mf_small
    del mol, mf, mf_small


def _sigma(fn, frozen=None, orbs=None, fullsigma=False, outcore=False):
    '''Imaginary-axis self-energy from upstream get_sigma / get_sigma_outcore or the lowmem one.'''
    gw = GWAC(mf, frozen=frozen)
    gw.orbs = orbs
    gw.fullsigma = fullsigma
    gw.verbose = 0
    gw.initialize_df()
    set_frozen_orbs(gw)
    e = _mo_energy_without_core(gw, gw.mo_energy)
    c = _mo_without_core(gw, gw.mo_coeff)
    qf, qw = _get_scaled_legendre_roots(gw.nw)
    ev = gw.setup_evaluation_grid(fallback_freqs=qf, fallback_wts=qw)
    ef = gw.get_ef(mo_energy=mf.mo_energy)
    L = None if outcore else gw.ao2mo(c)
    o = gw.orbs_frz
    if fn == 'upstream':
        return gw_ac.get_sigma(gw, o, L, qf, qw, ef, e, mo_coeff=c, iw_cutoff=5.0, eval_freqs=ev,
                               fullsigma=fullsigma)[0]
    if fn == 'upstream_outcore':
        return gw_ac.get_sigma_outcore(gw, o, qf, qw, ef, e, c, iw_cutoff=5.0, eval_freqs=ev,
                                       fullsigma=fullsigma)[0]
    return get_sigma_lowmem(gw, o, qf, qw, ef, e, c, iw_cutoff=5.0, eval_freqs=ev,
                            fullsigma=fullsigma, Lpq=L)[0]


class KnownValuesGWAC(unittest.TestCase):
    def setUp(self):
        # tiny blocks and frequency batches of three, so that every blocked loop is exercised
        self._mem = gw_ac_lowmem._mem_words
        gw_ac_lowmem._mem_words = lambda gw, reserve=0: 6 * gw.with_df.get_naoaux()**2 + 7

    def tearDown(self):
        gw_ac_lowmem._mem_words = self._mem

    def _check(self, **kw):
        ref = _sigma('upstream', **kw)
        scale = abs(ref).max()
        self.assertLess(abs(_sigma('lowmem', **kw) - ref).max() / scale, 1e-12)
        self.assertLess(abs(_sigma('lowmem', outcore=True, **kw) - ref).max() / scale, 1e-12)

    def test_sigma_all_orbs(self):
        self._check()

    def test_sigma_noncontiguous_orbs(self):
        self._check(orbs=[0, 3, 4, 5, 9])

    def test_sigma_frozen(self):
        self._check(frozen=1, orbs=list(range(2, 8)))

    def test_sigma_fullsigma(self):
        # out of core this crashes upstream unless segsize = 1
        self._check(fullsigma=True, orbs=list(range(1, 9)))

    def test_sigma_vs_upstream_outcore(self):
        ref = _sigma('upstream_outcore', orbs=[0, 3, 4, 5, 9], outcore=True)
        low = _sigma('lowmem', orbs=[0, 3, 4, 5, 9], outcore=True)
        self.assertLess(abs(low - ref).max() / abs(ref).max(), 1e-12)

    def test_kernel(self):
        # two-pole AC: Pade continuation amplifies rounding-level changes of sigma(iw)
        for kw in ({}, {'frozen': 1}, {'outcore': True}):
            a, b = GWAC(mf), GWACLowMem(mf)
            for g in (a, b):
                g.orbs = list(range(1, 9))
                g.ac = 'twopole'
                g.verbose = 0
                for k, v in kw.items():
                    setattr(g, k, v)
            a.kernel()
            b.kernel()
            self.assertAlmostEqual(abs(a.mo_energy[1:9] - b.mo_energy[1:9]).max(), 0, 9)

    def test_auto_outcore(self):
        a = GWAC(mf)
        b = GWACLowMem(mf)
        for g in (a, b):
            g.orbs = list(range(2, 7))
            g.ac = 'twopole'
            g.verbose = 0
        b.max_memory = lib.current_memory()[0] + 1e-4
        a.kernel()
        b.kernel()
        self.assertTrue(b.outcore)
        self.assertAlmostEqual(abs(a.mo_energy[2:7] - b.mo_energy[2:7]).max(), 0, 9)

    def test_exchange_direct(self):
        a, b = GWAC(mf), GWACLowMem(mf)
        self.assertAlmostEqual(abs(a.get_sigma_exchange(mf.mo_coeff) - b.get_sigma_exchange(mf.mo_coeff)).max(), 0, 9)


class KnownValuesGWCD(unittest.TestCase):
    def _run(self, cls, orbs, frozen=None, **kw):
        g = cls(mf, frozen=frozen)
        g.orbs = orbs
        g.verbose = 0
        for k, v in kw.items():
            setattr(g, k, v)
        g.kernel()
        return g.mo_energy

    def test_vs_upstream(self):
        # upstream GWCD is correct for orbs = range(n), frozen = None
        ref = self._run(GWCD, list(range(0, 7)))
        low = self._run(GWCDLowMem, list(range(0, 7)))
        self.assertAlmostEqual(abs(ref[:7] - low[:7]).max(), 0, 10)

    def test_orbs_subsets(self):
        full = self._run(GWCDLowMem, list(range(0, 8)))
        for orbs in ([0, 2, 4, 5, 7], list(range(2, 7))):     # both crash upstream
            sub = self._run(GWCDLowMem, orbs)
            self.assertAlmostEqual(abs(sub[orbs] - full[orbs]).max(), 0, 10)

    def test_frozen(self):
        # canonical CD reference with frozen core: fsCD with every residue evaluated exactly
        low = self._run(GWCDLowMem, list(range(1, 8)), frozen=1)
        ref = self._run(GWFSCD, list(range(1, 8)), frozen=1, fs_exact_cutoff=1e9)
        self.assertAlmostEqual(abs(low[1:8] - ref[1:8]).max(), 0, 9)


class KnownValuesSelfConsistent(unittest.TestCase):
    '''evGW / qsGW lowmem against upstream (two-pole AC, see KnownValuesGWAC.test_kernel).'''

    def _pair(self, ref_cls, low_cls, frozen=None, scf_obj=None, **kw):
        out = []
        for cls in (ref_cls, low_cls):
            g = cls(scf_obj or mf, frozen=frozen)
            g.ac = 'twopole'
            g.verbose = 0
            for k, v in kw.items():
                setattr(g, k, v)
            g.kernel()
            out.append(g.mo_energy)
        return abs(out[0] - out[1]).max()

    def test_evgw(self):
        for kw in ({}, {'W0': True}, {'frozen': 1}, {'outcore': True}, {'vhf_df': True}):
            frozen = kw.pop('frozen', None)
            self.assertAlmostEqual(self._pair(EVGW, EVGWLowMem, frozen=frozen, orbs=list(range(2, 8)), **kw), 0, 10)

    def test_evgw_preset_lpq(self):
        # upstream evgw.kernel fails (unbound local Lpq) when gw.Lpq is set beforehand
        ref = EVGW(mf)
        low = EVGWLowMem(mf)
        for g in (ref, low):
            g.ac = 'twopole'
            g.verbose = 0
            g.orbs = list(range(2, 8))
        low.initialize_df()
        low.Lpq = low.ao2mo(low.mo_coeff)
        ref.kernel()
        low.kernel()
        self.assertAlmostEqual(abs(ref.mo_energy - low.mo_energy).max(), 0, 10)

    def _qsgw_pair(self, frozen=None, **kw):
        '''Upstream QSGW and QSGWLowMem with the same self-energy function (upstream in-core
        get_sigma), single-threaded: a check of the kernel logic that is independent of the
        analytic continuation.  Comparing complete runs is not reliable: the two-pole fits do
        not converge for the core and the highest virtual and the Pade continuation amplifies
        rounding, so rounding-level differences of sigma(iw) (get_sigma_lowmem is tested
        against upstream separately) can move the QP energies by up to ~1e-4 Ha in rare runs.'''
        from pyscf.gw import qsgw_lowmem

        def upstream_sigma(gw, orbs, qf, qw, ef, e, c, mo_occ=None, iw_cutoff=None, eval_freqs=None,
                           fullsigma=False, Lpq=None):
            if Lpq is None:
                Lpq = GWAC.ao2mo(gw, c)
            return gw_ac.get_sigma(gw, orbs, Lpq, qf, qw, ef, e, mo_coeff=c, mo_occ=mo_occ,
                                   iw_cutoff=iw_cutoff, eval_freqs=eval_freqs, fullsigma=fullsigma)

        ref = QSGW(mf_small, frozen=frozen)
        low = QSGWLowMem(mf_small, frozen=frozen)
        if frozen:
            # upstream in core transforms the integrals with all orbitals but indexes them in
            # the frozen space (its out-of-core fullsigma path fails for any segsize); reference:
            # upstream with only the transformation restricted to the frozen-space orbitals
            ref.ao2mo = lambda mo_coeff=None: GWAC.ao2mo(ref, _mo_without_core(ref, mo_coeff))
        orig = qsgw_lowmem.get_sigma_lowmem
        qsgw_lowmem.get_sigma_lowmem = upstream_sigma
        try:
            with lib.with_omp_threads(1):
                for g in (ref, low):
                    # Pade is fast; its sensitivity to rounding does not matter here since both
                    # kernels get bit-identical self-energies
                    g.ac = 'pade'
                    g.verbose = 0
                    g.max_cycle = 3
                    for k, v in kw.items():
                        setattr(g, k, v)
                    if g is ref:
                        g.outcore = False
                    g.kernel()
        finally:
            qsgw_lowmem.get_sigma_lowmem = orig
        return abs(ref.mo_energy - low.mo_energy).max()

    def test_qsgw(self):
        for kw in ({'mode': 'b'}, {'mode': 'a'}, {'mode': 'c'}, {'vhf_df': True}, {'outcore': True}):
            self.assertAlmostEqual(self._qsgw_pair(**kw), 0, 10)

    def test_qsgw_frozen(self):
        self.assertAlmostEqual(self._qsgw_pair(frozen=1), 0, 10)

    def _qsgw_low(self, scf_obj, **kw):
        g = QSGWLowMem(scf_obj)
        g.ac = 'pade'
        g.verbose = 0
        g.max_cycle = 2
        for k, v in kw.items():
            setattr(g, k, v)
        with lib.with_omp_threads(1):
            g.kernel()
        return g

    def test_qsgw_scissor(self):
        nocc = mf_small.mol.nelectron // 2
        nmo = mf_small.mo_coeff.shape[1]
        # no window: no effect
        ref = self._qsgw_low(mf_small)
        for sc in ('hf', 'mf'):
            self.assertAlmostEqual(abs(self._qsgw_low(mf_small, scissor=sc).mo_energy - ref.mo_energy).max(), 0, 12)
        # window: outside orbitals get only a diagonal element, copied from a window orbital
        # of the same kind ('hf')
        win = list(range(nocc - 2, nocc + 2))
        g = self._qsgw_low(mf_small, orbs=win, scissor='hf')
        S = g.sigma_static
        out = [p for p in range(nmo) if p not in win]
        offdiag = S[out] - np.diag(np.diag(S))[out]
        self.assertAlmostEqual(abs(offdiag).max(), 0, 14)
        for p in out:
            same_kind = [q for q in win if (q < nocc) == (p < nocc)]
            self.assertAlmostEqual(min(abs(S[p, p] - S[q, q]) for q in same_kind), 0, 14)
        self.assertTrue(abs(np.diag(S)[out]).min() > 0)
        # Hartree-Fock reference: 'mf' and 'hf' coincide
        mhf = scf.RHF(mf_small.mol).density_fit().run()
        a = self._qsgw_low(mhf, orbs=win, scissor='hf')
        b = self._qsgw_low(mhf, orbs=win, scissor='mf')
        self.assertAlmostEqual(abs(a.mo_energy - b.mo_energy).max(), 0, 10)
        with self.assertRaises(ValueError):
            self._qsgw_low(mf_small, orbs=win, scissor='bogus')


class KnownValuesBSE(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gw = GWAC(mf)
        cls.gw.ac = 'twopole'
        cls.gw.verbose = 0
        cls.gw.kernel()

    def _cmp(self, multi, **kw):
        a, b = BSE(self.gw), BSELowMem(self.gw)
        for x in (a, b):
            x.verbose = 0
            x.nroot = 5
            for k, v in kw.items():
                setattr(x, k, v)
        # Davidson converges to residue_thresh = 1e-8; different rounding (threaded BLAS/einsum
        # reductions on differently laid-out but identical blocks) can change the final
        # iterates at the 1e-11 level
        self.assertAlmostEqual(abs(a.kernel(multi)[0] - b.kernel(multi)[0]).max(), 0, 9)

    def test_singlet_triplet_tda(self):
        self._cmp('s')
        self._cmp('t')
        self._cmp('s', TDA=True)

    def test_subspace_trial(self):
        self._cmp('s', trial='subspace', nocc_sub=3, nvir_sub=6)

    def test_full_diagonalization(self):
        a, b = BSE(self.gw), BSELowMem(self.gw)
        for x in (a, b):
            x.verbose = 0
        self.assertAlmostEqual(abs(a.full_diagonalization('s')[0] - b.full_diagonalization('s')[0]).max(), 0, 11)

    def test_blocks_from_df(self):
        gw = self.gw
        L = gw.Lpq
        try:
            gw.Lpq = None
            b = BSELowMem(gw)
        finally:
            gw.Lpq = L
        full = np.asarray(L)
        self.assertAlmostEqual(abs(np.asarray(b.Lpq)[0] - full).max(), 0, 12)
        self.assertAlmostEqual(abs(b.Lpq[0, :, 2:9, 3:11] - full[:, 2:9, 3:11]).max(), 0, 12)
        # stored blocks are returned without copies
        o = int(b.nocc[0])
        self.assertIs(b.Lpq[0][:, :o, o:], b.Lpq.spins[0].b['ov'])


if __name__ == '__main__':
    print('Full tests for low-memory GW-AC, GW-CD and BSE')
    unittest.main()
