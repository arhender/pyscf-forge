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
Linear, quadratic and cubic response for restricted TDHF / TDDFT (adiabatic, closed shell, real
orbitals): polarizabilities, first and second hyperpolarizabilities, two- and three-photon
absorption.

Theory
------
S. M. Parker, D. Rappoport, F. Furche, J. Chem. Theory Comput. 14, 807 (2018), Sec. 1.

The general equation-of-motion machinery (first- and second-order equations, quadratic
response function, two-photon residues) is in pyscf.tdscf.qr_base.  The TDDFT-specific parts are

    F[D] = F0 + V(t) + G[D - D0] + 1/2 g_xc[D - D0, D - D0] + ...
    G[D]_pq   = 2 (pq|rs) D_rs - c_x (pr|sq) D_rs + f_xc[rho(2D)]_pq             (Parker Eq. 30)
    g_xc[B,C] = int phi_p phi_q (d^3 E_xc / d rho^3)(rho0) rho(2B) rho(2C)       (Parker Eqs. 31-32)

(range-separated exchange is handled by pyscf's response function).  The second-order kernel
is G[K] + g_xc[D1^B, D1^C]; its symmetric (A+B) and skew-symmetric (A-B) channels are
r_ai +/- r_ia, reproducing Parker Eq. 28: G[K]_ai + G[K]_ia = 2 H+[K+],
G[K]_ai - G[K]_ia = 2 H-[K-], and g_xc only enters the symmetric channel.

Cubic response additionally needs the third-order XC kernel (fourth functional derivative,
B. Jansik et al., J. Chem. Phys. 122, 054107 (2005)),

    h_xc[B,C,D] = int phi_p phi_q (d^4 E_xc / d rho^4)(rho0) rho(2B) rho(2C) rho(2D),

which is controlled by the same `hyperkernel` flag as g_xc.
'''

import numpy as np

from pyscf import lib, scf
from pyscf.tdscf.qr_base import (QuadraticResponseBase, beta_parallel, gamma_parallel,  # noqa: F401
                                 _tpa_strength, _tpa3_strength)

einsum = lib.einsum


class TDDFTQuadraticResponse(QuadraticResponseBase):
    '''Linear and quadratic response solver for restricted TDHF / adiabatic TDDFT.

    Attributes (in addition to those of QuadraticResponseBase):
        hyperkernel : bool
            Include the exchange-correlation hyperkernel g_xc (third functional derivative) and,
            in cubic response, the third-order kernel h_xc (fourth functional derivative).
            Required for consistency with finite-field derivatives.  Default True.
    '''

    _label = 'TDDFT'

    def __init__(self, mf):
        if not isinstance(mf, scf.hf.RHF) or isinstance(mf, (scf.uhf.UHF, scf.rohf.ROHF)):
            raise NotImplementedError('TDDFT quadratic response is implemented for RHF/RKS references only.')
        if mf.mo_coeff is None:
            raise RuntimeError('The mean-field object must be converged before quadratic response.')
        QuadraticResponseBase.__init__(
            self, mf.mol, mf.mo_energy, mf.mo_coeff, np.count_nonzero(np.asarray(mf.mo_occ) > 0),
            verbose=mf.verbose, stdout=mf.stdout, max_memory=mf.max_memory)
        self._scf = mf
        self.mo_occ = np.asarray(mf.mo_occ)
        self.hyperkernel = True
        self._vind = None
        self._xc = None
        if not self.is_dft:
            self._label = 'TDHF'

    @property
    def is_dft(self):
        return isinstance(self._scf, scf.hf.KohnShamDFT)

    def _dump_method_flags(self, log):
        log.info('reference = %s', self._scf.__class__.__name__)
        if self.is_dft:
            log.info('xc = %s', self._scf.xc)
            log.info('xc hyperkernel = %s', self.hyperkernel)

    def build(self):
        '''Set up the response function and, for DFT, the XC hyperkernel on the grid.'''
        mf = self._scf
        # singlet=None: response of the closed-shell Fock matrix to a total density change
        # (2J - c_x K + f_xc), valid for non-symmetric densities with hermi=0
        self._vind = mf.gen_response(self.mo_coeff, self.mo_occ, singlet=None, hermi=0)
        self._xc = None
        if self.is_dft:
            if mf.do_nlc():
                raise NotImplementedError('Non-local correlation (NLC) is not supported.')
            ni = mf._numint
            xctype = ni._xc_type(mf.xc)
            if self.hyperkernel and xctype in ('LDA', 'GGA', 'MGGA'):
                ni.libxc.test_deriv_order(mf.xc, 3, raise_error=True)
                grids = getattr(mf, 'second_grids', None) or mf.grids
                if grids.coords is None:
                    grids.build(with_non0tab=True)
                rho0, vxc, fxc = ni.cache_xc_kernel(self.mol, grids, mf.xc, self.mo_coeff, self.mo_occ, 0)
                kxc = ni.eval_xc_eff(mf.xc, rho0, deriv=3, xctype=xctype, spin=0)[3]
                self._xc = {'ni': ni, 'grids': grids, 'xctype': xctype,
                            'rho0': rho0, 'vxc': vxc, 'kxc': kxc}
        return QuadraticResponseBase.build(self)

    def get_kernel(self, dm):
        '''Kernel G[D] (Parker Eq. 30) in the MO basis for per-spin, possibly non-symmetric D.'''
        self._check_built()
        dm = np.asarray(dm)
        single = dm.ndim == 2
        if single:
            dm = dm[None]
        v = self._vind(2.0 * self._mo2ao(dm))
        out = self._ao2mo(np.asarray(v).reshape(dm.shape[0], *v.shape[-2:]))
        return out[0] if single else out

    def _rho_on_grid(self, dm_ao):
        '''Density (and derivatives) of a total AO density matrix on the full grid.'''
        from pyscf.dft import numint
        xc = self._xc
        ni, grids, xctype = xc['ni'], xc['grids'], xc['xctype']
        if xctype == 'LDA':
            ao_deriv = 0
        elif xctype == 'GGA':
            ao_deriv = 1
        else:
            ao_deriv = 2 if numint.MGGA_DENSITY_LAPL else 1
        make_rho, nset, nao = ni._gen_rho_evaluator(self.mol, dm_ao, 0, False, grids)
        rho = []
        for ao, mask, weight, coords in ni.block_loop(self.mol, grids, nao, ao_deriv,
                                                       max_memory=self.max_memory):
            rho.append(np.atleast_2d(make_rho(0, ao, mask, xctype)))
        return np.concatenate(rho, axis=-1)

    def get_hyperkernel(self, dm_b, dm_c):
        '''XC hyperkernel g_xc[D_b, D_c] in the MO basis (Parker Eq. 31) for per-spin D_b, D_c.'''
        self._check_built()
        if self._xc is None:
            return np.zeros((self.nmo, self.nmo))
        xc = self._xc
        rho_c = self._rho_on_grid(2.0 * self._mo2ao(np.asarray(dm_c)[None])[0])
        nvar = rho_c.shape[0]
        kxc = xc['kxc'].reshape(nvar, nvar, nvar, -1)
        fxc_eff = einsum('xyzg,zg->xyg', kxc, rho_c)
        dmb_ao = 2.0 * self._mo2ao(np.asarray(dm_b)[None])[0]
        v = xc['ni'].nr_rks_fxc(self.mol, xc['grids'], self._scf.xc, None, dmb_ao, 0, 0,
                                xc['rho0'], xc['vxc'], fxc_eff, max_memory=self.max_memory)
        return self._ao2mo(np.asarray(v).reshape(1, *v.shape[-2:]))[0]

    def get_third_kernel(self, dm_b, dm_c, dm_d):
        '''Third-order XC kernel h_xc[D_b, D_c, D_d] in the MO basis for per-spin densities:
        int phi_p phi_q (d^4 E_xc / d rho^4)(rho0) rho(2B) rho(2C) rho(2D).

        The fourth derivative of the functional is evaluated on first use and cached.'''
        self._check_built()
        if self._xc is None:
            return np.zeros((self.nmo, self.nmo))
        xc = self._xc
        if 'lxc' not in xc:
            xc['ni'].libxc.test_deriv_order(self._scf.xc, 4, raise_error=True)
            xc['lxc'] = xc['ni'].eval_xc_eff(self._scf.xc, xc['rho0'], deriv=4,
                                             xctype=xc['xctype'], spin=0)[4]
        rho_c = self._rho_on_grid(2.0 * self._mo2ao(np.asarray(dm_c)[None])[0])
        rho_d = self._rho_on_grid(2.0 * self._mo2ao(np.asarray(dm_d)[None])[0])
        nvar = rho_c.shape[0]
        lxc = xc['lxc'].reshape(nvar, nvar, nvar, nvar, -1)
        fxc_eff = einsum('xyzwg,zg,wg->xyg', lxc, rho_c, rho_d)
        dmb_ao = 2.0 * self._mo2ao(np.asarray(dm_b)[None])[0]
        v = xc['ni'].nr_rks_fxc(self.mol, xc['grids'], self._scf.xc, None, dmb_ao, 0, 0,
                                xc['rho0'], xc['vxc'], fxc_eff, max_memory=self.max_memory)
        return self._ao2mo(np.asarray(v).reshape(1, *v.shape[-2:]))[0]

    def _hyperkernel(self, Db, Dc):
        if self.hyperkernel and self._xc is not None:
            return self.get_hyperkernel(Db, Dc)
        return None

    def _third_kernel(self, Db, Dc, Dd):
        if self.hyperkernel and self._xc is not None:
            return self.get_third_kernel(Db, Dc, Dd)
        return None

    def _excited_states_large(self, nstates):
        td = self._scf.TDDFT() if self.is_dft else self._scf.TDHF()
        td.nstates = nstates
        td.conv_tol = min(td.conv_tol, 1e-9)
        td.kernel()
        X = np.array([xy[0].ravel() for xy in td.xy])
        Y = np.array([xy[1].ravel() for xy in td.xy])
        return td.e, X, Y


QRTDDFT = TDDFTQuadraticResponse
