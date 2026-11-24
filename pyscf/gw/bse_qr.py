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
Linear and quadratic response for the static-screened Bethe-Salpeter equation (BSE@GW).

Restricted closed-shell, singlet perturbations, real orbitals.

Theory
------
Follows N. Rauwolf, W. Klopper, C. Holzer, J. Chem. Phys. 160, 061101 (2024) and the
summary in C. Holzer, Y. J. Franzke, arXiv:2502.08753 (Eqs. 16-32):

* The BSE kernel is the static screened kernel Xi(Omega) ~ W(0) used by pyscf.gw.bse.
* The BSE hyperkernel (second derivative of the self-energy w.r.t. G) is neglected.

With these approximations the BSE response equations are those of a time-dependent
mean-field theory with a Fock operator that is linear in the density,

    F[D] = diag(eps_QP) + V(t) + G[D - D0],
    G[D]_pq = 2 sum_rs (pq|rs) D_rs - sum_rs W(pr|sq) D_rs,

where eps_QP are GW quasiparticle energies and W is the statically screened Coulomb
interaction.  All working equations below follow from the equation of motion
i dD/dt = [F, D] together with idempotency D^2 = D (per-spin density, D0 = occupied projector).

First order (operator V at frequency w):
    (A - w) X + B Y = -V_ai,      B X + (A + w) Y = -V_ia,
    D1_ai = X_ia,  D1_ia = Y_ia,  F1 = V + G[D1]          (U/V of Eqs. 29-30)

Second order (perturbations B at w1, C at w2, w = w1 + w2):
    K_ij = -sum_a (Y^B_ia X^C_ja + Y^C_ia X^B_ja)            (Eq. 25)
    K_ab =  sum_i (X^B_ia Y^C_ib + X^C_ia Y^B_ib)            (Eq. 26)
    Comm = [F1^B, D1^C] + [F1^C, D1^B]                        (Eqs. 27-28)
    r_ai = Comm_ai + G[K]_ai,    r_ia = -Comm_ia + G[K]_ia
    (A - w) X^BC + B Y^BC = -r_ai,   B X^BC + (A + w) Y^BC = -r_ia

G[K] (the kernel acting on the occupied-occupied/virtual-virtual blocks of the second-order
density) corresponds to the H+[K+] and H-[K-] terms of Eqs. 22-23 of Rauwolf et al.  It is
required for the exact TDHF limit and for the intrinsic permutation symmetry of the quadratic
response function.

Relation to the printed equations of Rauwolf et al. (verified numerically to 1e-14):
M = 2 sym(F1), N = -2 antisym(F1), K+/K- = symmetric/antisymmetric parts of K, and the
2PA amplitude of Eq. 27 equals T^N / sqrt(2) (T^N includes the closed-shell spin factor).
Agreement requires reading N_ij as N_ji in the occupied sums of Eqs. 22-23 and flipping the
sign of the H-[K-] term (H- taken at ai rather than ia); both are forced by the TDHF limit.
Eq. 7 should also contain the orbital-energy difference, and Eqs. 3-4/20-21 are shorthand
for the coupled equations solved in solve().
Both corrections agree with the TDDFT quadratic-response right-hand side of S. M. Parker,
D. Rappoport, F. Furche, J. Chem. Theory Comput. 14, 807 (2018), Eq. 28 (U_ji in the occupied
sums, -H-_ia[K-]), which reduces to the BSE expressions with f_xc = g_xc = 0 and the exchange
integrals replaced by W.  The static limit is verified against finite-field derivatives of the
fixed-W model F = diag(eps_QP) + G[D - D0] + F.r (test_screened_finite_field).

The method-independent machinery (linear solver, second-order right-hand side, quadratic
response function, two-photon residues) lives in pyscf.tdscf.qr_base and is shared with
TDHF/TDDFT (pyscf.tdscf.quadratic_response); this module supplies the BSE kernel.

Quadratic response function (spin-summed density, rho = 2 D):
    <<A; B, C>>_{w1,w2} = 2 [ Tr(A K) + Tr(A D2_ov/vo) ]
                        = 2 [ Tr(A K) + z_A(w1 + w2) . r^BC ]
where z_A(w) = (X_A, Y_A) is the first-order response of A at frequency w (the linear
equations are symmetric), so no second-order equations need to be solved.

Conventions
-----------
Perturbation H' = F . r (dipole operator mu = -r), Taylor-series convention
mu(F) = mu0 + alpha F + 1/2 beta F F, i.e.
    alpha_xy = -<<r_x; r_y>>,   beta_xyz(-w1-w2; w1, w2) = -<<r_x; r_y, r_z>>.

Two-photon transition moments come from the residue of the quadratic response function,
    lim_{w1+w2 -> Omega_N} (w1 + w2 - Omega_N) <<A; B, C>> = <0|A|N> T^N_BC,
    T^N_BC = sqrt(2) z_N . r^BC(w1, w2),
with z_N the spin-adapted BSE eigenvector normalized to X.X - Y.Y = 1.
'''

import numpy as np

from pyscf import lib
from pyscf.lib import logger
from pyscf.gw.bse import BSE, _bse_contraction, _get_lpq_bar
from pyscf.gw.qp_shift import scissor_shift
from pyscf.tdscf.qr_base import QuadraticResponseBase, beta_parallel, _tpa_strength  # noqa: F401

einsum = lib.einsum


class BSEQuadraticResponse(QuadraticResponseBase):
    '''Linear and quadratic response solver for restricted static-screened BSE@GW.

    Attributes (in addition to those of QuadraticResponseBase):
        screening : bool
            Use the statically screened W in the exchange-type kernel (BSE).  If False the
            bare Coulomb interaction is used, which gives (DF-)TDHF when the reference is a
            Hartree-Fock object.  Default True.
        xc_hyperkernel : bool
            Diagnostic option, not part of the BSE theory: add the exchange-correlation
            hyperkernel g_xc[D1^B, D1^C] of the Kohn-Sham reference functional to the
            second-order kernel, as in TDDFT quadratic response.  Default False.
    '''

    _label = 'BSE'

    def __init__(self, obj, screening=True, scissor=True):
        '''Initialize from a BSE object, a GW object (GWAC), or a restricted mean-field object.

        Passing a mean-field object uses its orbital energies instead of GW quasiparticle
        energies (useful for testing: with screening=False and an RHF reference this is TDHF).

        If a GW object only corrected the orbitals in gw.orbs, the remaining orbitals are
        filled in by a scissor shift (see pyscf.gw.qp_shift.scissor_shift) unless
        scissor=False.  The GW object itself is not modified.  A BSE object must already
        carry complete quasiparticle energies; apply scissor_shift to the GW object first.
        '''
        if isinstance(obj, BSE):
            bse = obj
        elif hasattr(obj, '_scf') and hasattr(obj, 'mo_energy'):
            bse = _bse_from_gw(obj, scissor)
        elif hasattr(obj, 'mo_energy') and hasattr(obj, 'mol'):
            bse = _bse_from_scf(obj)
        else:
            raise TypeError('Expected a BSE, GW or restricted mean-field object, got %s' % type(obj))

        if bse.nspin != 1:
            raise NotImplementedError('BSE quadratic response is implemented for restricted references only.')

        QuadraticResponseBase.__init__(
            self, bse.mol, bse.mo_energy[0], bse.mo_coeff[0], bse.nocc[0],
            verbose=bse.verbose, stdout=bse.stdout,
            max_memory=getattr(bse.mf, 'max_memory', bse.mol.max_memory))
        self._bse = bse
        self.Lpq = np.asarray(bse.Lpq[0])
        if np.count_nonzero(self.mo_energy == 0.0) > 0:
            raise RuntimeError(
                'Some quasiparticle energies are exactly zero: GW was probably restricted to gw.orbs. '
                'Apply pyscf.gw.qp_shift.scissor_shift(gw, inplace=True) before building BSE, '
                'or pass the GW object (scissor=True) instead.')
        if self.Lpq.shape[1] != self.nmo or self.mo_coeff.shape[1] != self.nmo:
            raise NotImplementedError('Frozen orbitals are not supported in BSE quadratic response.')

        self.screening = screening
        self.xc_hyperkernel = False
        self._Lbar = None
        self._xc_qr = None

    def _dump_method_flags(self, log):
        log.info('naux = %d', self.Lpq.shape[0])
        log.info('screened kernel (BSE) = %s', self.screening)
        log.info('BSE hyperkernel neglected (static screening)')
        if self.xc_hyperkernel:
            log.info('DIAGNOSTIC: reference XC hyperkernel added to second-order kernel (%s)',
                     getattr(self._bse.mf, 'xc', None))

    def build(self):
        '''Precompute the (screened) DF intermediates used by the kernel contractions.'''
        o = self.nocc
        L = self.Lpq
        if self.screening:
            self._Lbar = _get_lpq_bar(nocc=[o], mo_energy=self.mo_energy[None], Lpq=L[None])[0]
        else:
            self._Lbar = L
        Lbar = self._Lbar
        # layouts expected by pyscf.gw.bse._bse_contraction
        self._Lia = [np.ascontiguousarray(L[:, :o, o:])]
        self._Laa = [np.ascontiguousarray(L[:, o:, o:])]
        self._Lii_bar = [np.ascontiguousarray(Lbar[:, :o, :o].transpose(1, 0, 2))]
        self._Lia_bar = [np.ascontiguousarray(Lbar[:, :o, o:].transpose(1, 0, 2))]
        self._xc_qr = None
        if self.xc_hyperkernel:
            self._xc_qr = _reference_xc_hyperkernel(self)
        return QuadraticResponseBase.build(self)

    def _second_order_kernel(self, K, D1b, D1c):
        G2 = self.get_kernel(K)
        if self.xc_hyperkernel:
            if self._xc_qr is None:
                self._xc_qr = _reference_xc_hyperkernel(self)
            G2 = G2 + self._xc_qr.get_hyperkernel(D1b, D1c)
        return G2

    def get_kernel(self, dm):
        '''Kernel G[D] in the full MO space for one or several (possibly non-symmetric) D.

        G[D]_pq = 2 sum_rs (pq|rs) D_rs - sum_rs W(pr|sq) D_rs
        '''
        self._check_built()
        dm = np.asarray(dm)
        single = dm.ndim == 2
        if single:
            dm = dm[None]
        L = self.Lpq
        Lbar = self._Lbar
        rho = einsum('Prs,nrs->nP', L, dm)
        vj = einsum('Ppq,nP->npq', L, rho)
        tmp = einsum('Ppr,nrs->nPps', L, dm)
        vk = einsum('nPps,Psq->npq', tmp, Lbar)
        out = 2.0 * vj - vk
        return out[0] if single else out

    def apb_amb_prod(self, zp, zm):
        '''Return (A+B) zp and (A-B) zm for vectors in the (occ, vir) flattened space.'''
        self._check_built()
        o = self.nocc
        tri = np.vstack([zp.ravel(), zm.ravel()])
        apb, amb, _ = _bse_contraction(
            's', [o], self.mo_energy[None], self._Lia, self._Laa,
            self._Lii_bar, self._Lia_bar, tri, TDA=False)
        return apb[0], amb[1]

    def _lr_matvec(self, omega, X, Y):
        '''[[A - w, B], [B, A + w]] (X, Y) via the occupied-virtual BSE contraction of pyscf.gw.bse.'''
        ap, am = self.apb_amb_prod(X + Y, X - Y)
        ap = ap.reshape(X.shape)
        am = am.reshape(X.shape)
        return 0.5 * (ap + am) - omega * X, 0.5 * (ap - am) + omega * Y

    def get_dense_apb_amb(self):
        '''Dense A+B and A-B matrices (dimension nocc*nvir).'''
        if self._apb is None:
            self._check_built()
            o = self.nocc
            dim = o * (self.nmo - o)
            apb, amb, _ = _bse_contraction(
                's', [o], self.mo_energy[None], self._Lia, self._Laa,
                self._Lii_bar, self._Lia_bar, np.eye(dim), TDA=False)
            self._set_dense(apb, amb)
        return self._apb, self._amb

    def _excited_states_large(self, nstates):
        if not self.screening:
            raise NotImplementedError('Use solver="direct" for unscreened excited states.')
        bse = self._bse
        bse.nroot = nstates
        bse.max_vec = max(bse.max_vec, 12 * nstates)
        exci, X_vec, Y_vec = bse.kernel('s')
        dim = self.nocc * (self.nmo - self.nocc)
        return exci, X_vec[0].reshape(-1, dim), Y_vec[0].reshape(-1, dim)


def _reference_xc_hyperkernel(qr):
    '''TDDFT helper evaluating the reference functional's XC hyperkernel on the BSE orbitals.'''
    from pyscf.tdscf.quadratic_response import TDDFTQuadraticResponse
    mf = qr._bse.mf
    if not np.allclose(np.asarray(mf.mo_coeff), qr.mo_coeff):
        raise RuntimeError('xc_hyperkernel requires the BSE orbitals to be the reference KS orbitals.')
    helper = TDDFTQuadraticResponse(mf)
    helper.verbose = 0
    helper.build()
    if helper._xc is None:
        logger.warn(qr, 'xc_hyperkernel: reference has no semilocal XC part; the term is zero.')
    return helper


def _bse_from_gw(gw, scissor=True):
    if getattr(gw, 'frozen', None) is not None:
        raise NotImplementedError('Frozen orbitals are not supported in BSE quadratic response.')
    if getattr(gw, 'Lpq', None) is None and getattr(gw, 'with_df', None) is None:
        gw.initialize_df(auxbasis=getattr(gw, 'auxbasis', None))
    bse = BSE(gw)
    if scissor and np.asarray(gw.mo_energy).ndim == 1:
        bse.mo_energy = scissor_shift(gw)[None, :]
    return bse


def _bse_from_scf(mf):
    from pyscf.gw.gw_ac import GWAC
    if np.asarray(mf.mo_energy).ndim != 1:
        raise NotImplementedError('BSE quadratic response is implemented for restricted references only.')
    gw = GWAC(mf)
    gw.initialize_df()
    return BSE(gw)


QRBSE = BSEQuadraticResponse
