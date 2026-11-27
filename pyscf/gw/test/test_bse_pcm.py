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
from pyscf.gw.gw_fscd import GWFSCD
from pyscf.gw.bse_lowmem import BSELowMem
from pyscf.gw.bse_qr import BSEQuadraticResponse
from pyscf.gw.qp_shift import scissor_shift
from pyscf.gw import pcm_gw


def setUpModule():
    global mol, mf, orbs
    mol = gto.M(atom='O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692',
                basis='def2-svp', verbose=0)
    mf = dft.RKS(mol, xc='pbe0').PCM()
    mf.with_solvent.method = 'IEF-PCM'
    mf.with_solvent.eps = 78.3553
    mf.conv_tol = 1e-12
    mf.kernel()
    nocc = mol.nelectron // 2
    orbs = list(range(nocc - 2, nocc + 2))


def tearDownModule():
    global mol, mf
    del mol, mf


def _gw(scheme, eps_fast=None, **kwargs):
    gw = GWFSCD(mf, auxbasis='def2-svp-ri')
    gw.orbs = orbs        # the other orbitals are scissor-shifted
    for k, v in kwargs.items():
        setattr(gw, k, v)
    gw = pcm_gw.GW_PCM(gw, scheme=scheme, eps_fast=eps_fast)
    gw.kernel()
    scissor_shift(gw, inplace=True)
    return gw


def _bse(bse, nroot=4):
    bse.nroot = nroot
    bse.kernel('s')
    return np.asarray(bse.exci).ravel()[:nroot]


class KnownValues(unittest.TestCase):
    def test_eps1_identity(self):
        '''eps_fast = 1: the dressed BSE kernels equal the bare ones.'''
        gw = _gw('cohsex', 1.0)
        ref = _bse(BSELowMem(gw))
        self.assertAlmostEqual(abs(_bse(pcm_gw.make_bse(gw)) - ref).max(), 0, 8)
        a0 = BSEQuadraticResponse(BSELowMem(gw)).polarizability()
        a1 = BSEQuadraticResponse(pcm_gw.make_bse(gw)).polarizability()
        self.assertAlmostEqual(abs(a1 - a0).max(), 0, 6)

    def test_integral_paths(self):
        '''Dressed MO integrals (in-core gw.Lpq) and dressed DF object (lowmem_ints) agree.'''
        for scheme in ('cohsex', 'gw'):
            e_incore = _bse(pcm_gw.make_bse(_gw(scheme)))
            e_lowmem = _bse(pcm_gw.make_bse(_gw(scheme, lowmem_ints=True)))
            self.assertAlmostEqual(abs(e_incore - e_lowmem).max(), 0, 7)

    def test_solvent_response(self):
        '''The fast response lowers the singlets of water (the QP gap shrinks more than the
        electron-hole attraction weakens) and increases the polarizability.'''
        gw1 = _gw('cohsex', 1.0)
        gw = _gw('cohsex')
        e1 = _bse(BSELowMem(gw1))
        e = _bse(pcm_gw.make_bse(gw))
        self.assertTrue(np.all(e < e1))
        a1 = BSEQuadraticResponse(BSELowMem(gw1)).polarizability()
        a = BSEQuadraticResponse(pcm_gw.make_bse(gw)).polarizability()
        self.assertTrue(np.all(np.diag(a) > np.diag(a1)))


if __name__ == '__main__':
    print('Full Tests for BSE-PCM')
    unittest.main()
