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
from pyscf import gto, dft, df
from pyscf.ao2mo._ao2mo import nr_e2
from pyscf.gw.gw_ac import GWAC
from pyscf.gw.gw_ac_lowmem import GWACLowMem
from pyscf.gw.gw_cd_lowmem import GWCDLowMem
from pyscf.gw.gw_fscd import GWFSCD
from pyscf.gw.evgw_lowmem import EVGWLowMem
from pyscf.gw.evgw_fscd import EVGWFSCD
from pyscf.gw.bse_lowmem import transform_df_pairs
from pyscf.gw import pcm_gw

HA2EV = 27.211386245988


def setUpModule():
    global mol, mf, nocc, orbs, auxbasis
    mol = gto.M(atom='O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692',
                basis='def2-svp', verbose=0)
    mf = dft.RKS(mol, xc='pbe0').PCM()
    mf.with_solvent.method = 'IEF-PCM'
    mf.with_solvent.eps = 78.3553
    mf.conv_tol = 1e-12
    mf.kernel()
    nocc = mol.nelectron // 2
    orbs = [nocc - 2, nocc - 1, nocc, nocc + 1]
    auxbasis = 'def2-svp-ri'


def tearDownModule():
    global mol, mf
    del mol, mf


def _mo_lpq(with_df, mo):
    nmo = mo.shape[1]
    return nr_e2(with_df._cderi, np.asarray(mo, order='F'), (0, nmo, 0, nmo), aosym='s2').reshape(-1, nmo, nmo)


def _run(cls, scheme=None, eps_fast=None):
    gw = cls(mf, auxbasis=auxbasis)
    gw.orbs = orbs
    if scheme is not None:
        gw = pcm_gw.GW_PCM(gw, scheme=scheme, eps_fast=eps_fast)
    gw.kernel()
    return gw


class KnownValues(unittest.TestCase):
    def test_reaction_integrals(self):
        '''RI (pq|v_reac|rs) against the PCM response of pyscf.solvent (no RI).'''
        wdf = df.DF(mol, auxbasis=auxbasis).build()
        U, info = pcm_gw.get_reaction_metric(wdf, mf.with_solvent)
        self.assertAlmostEqual(info['pcm'].eps, 1.78, 12)
        C = mf.mo_coeff
        L = _mo_lpq(wdf, C)
        Lt = pcm_gw.dress_lpq(L, U)
        pcm = info['pcm']
        for p, q in ((nocc - 1, nocc - 1), (nocc - 2, nocc + 1)):
            ri = Lt[:, p, q] @ Lt[:, p, q] - L[:, p, q] @ L[:, p, q]
            dm = np.outer(C[:, p], C[:, q])
            dm = 0.5 * (dm + dm.T)
            exact = C[:, p] @ pcm._B_dot_x(dm[None])[0] @ C[:, q]
            self.assertLess(ri, 0)
            self.assertAlmostEqual(ri / exact, 1.0, 2)

        U1 = pcm_gw.get_reaction_metric(wdf, mf.with_solvent, eps_fast=1.0)[0]
        self.assertAlmostEqual(abs(U1 - np.eye(U1.shape[0])).max(), 0, 14)

    def test_dressed_df(self):
        wdf = df.DF(mol, auxbasis=auxbasis).build()
        U = pcm_gw.get_reaction_metric(wdf, mf.with_solvent)[0]
        C = mf.mo_coeff
        ref = pcm_gw.dress_lpq(_mo_lpq(wdf, C), U)
        incore = pcm_gw.make_dressed_df(wdf, U)
        self.assertAlmostEqual(abs(_mo_lpq(incore, C) - ref).max(), 0, 12)
        ondisk = pcm_gw.make_dressed_df(wdf, U, max_memory=1)
        self.assertIsInstance(ondisk._cderi, str)
        nmo = C.shape[1]
        blk = transform_df_pairs(ondisk, C, {'all': (slice(0, nmo), slice(0, nmo))}, 4000)['all']
        self.assertAlmostEqual(abs(blk - ref).max(), 0, 12)

    def test_eps1_identity(self):
        '''Without fast response (eps_fast = 1) every driver reproduces the plain run.'''
        for cls in (GWACLowMem, GWCDLowMem, GWFSCD, EVGWLowMem, EVGWFSCD):
            e0 = np.asarray(_run(cls).mo_energy)[orbs]
            for scheme in ('cohsex', 'gw'):
                e1 = np.asarray(_run(cls, scheme, 1.0).mo_energy)[orbs]
                self.assertAlmostEqual(abs(e1 - e0).max(), 0, 8, msg='%s %s' % (cls.__name__, scheme))

    def test_g0w0_shifts(self):
        '''Fast-response shifts (eV) of HOMO-1, HOMO, LUMO, LUMO+1: occupied levels up,
        virtual levels down.'''
        e0 = np.asarray(_run(GWACLowMem).mo_energy)[orbs]
        gw = _run(GWACLowMem, 'cohsex')
        d_coh = (np.asarray(gw.mo_energy)[orbs] - e0) * HA2EV
        self.assertAlmostEqual(abs(d_coh - gw.pcm_shift * HA2EV).max(), 0, 10)
        self.assertAlmostEqual(abs(d_coh - [1.5914, 1.6136, -1.9005, -1.8368]).max(), 0, 3)
        d_gw = (np.asarray(_run(GWACLowMem, 'gw').mo_energy)[orbs] - e0) * HA2EV
        self.assertAlmostEqual(abs(d_gw - [1.4227, 1.4508, -1.8708, -1.7907]).max(), 0, 3)
        # the same for contour deformation (a different frequency integration)
        d_cd = (np.asarray(_run(GWCDLowMem, 'gw').mo_energy)[orbs] - e0) * HA2EV
        self.assertAlmostEqual(abs(d_cd - d_gw).max(), 0, 3)

    def test_evgw_ham_hf(self):
        '''The static evGW Hamiltonian of a PCM reference keeps the ground-state solvent
        potential: ham_hf - Sigma^x + vxc is the mean-field Fock matrix.'''
        gw = EVGWLowMem(mf, auxbasis=auxbasis)
        C = mf.mo_coeff
        ham_hf, vk = gw.get_ham_hf(C)
        dm = mf.make_rdm1()
        vxc = C.T @ (mf.get_veff(dm=dm) - mf.get_j(dm=dm)) @ C
        self.assertAlmostEqual(abs(np.diag(ham_hf - vk + vxc) - mf.mo_energy).max(), 0, 6)

    def test_unsupported(self):
        with self.assertRaises(NotImplementedError):
            pcm_gw.GW_PCM(GWAC(mf))
        with self.assertRaises(ValueError):
            pcm_gw.GW_PCM(GWACLowMem(mf), scheme='bse')


if __name__ == '__main__':
    print('Full Tests for GW-PCM')
    unittest.main()
