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
import scipy.linalg
import scipy.sparse.linalg as spla

from pyscf import lib
from pyscf.lib import logger
from pyscf.gw.bse import BSE, _bse_contraction, _get_lpq_bar
from pyscf.gw.qp_shift import scissor_shift

einsum = lib.einsum


class BSEQuadraticResponse(lib.StreamObject):
    '''Linear and quadratic response solver for restricted static-screened BSE@GW.

    Attributes:
        screening : bool
            Use the statically screened W in the exchange-type kernel (BSE).  If False the
            bare Coulomb interaction is used, which gives (DF-)TDHF when the reference is a
            Hartree-Fock object.  Default True.
        solver : str
            'iterative' (GMRES with a diagonal preconditioner, default) or 'direct'
            (dense A+B / A-B matrices; small systems only).
        conv_tol : float
            Relative residual tolerance of the iterative linear solver.
        max_cycle : int
            Maximum number of GMRES iterations per right-hand side.
        gauge_origin : 3-tuple
            Origin of the dipole operator.  Results for neutral molecules do not depend on it.
    '''

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

        self._bse = bse
        self.mol = bse.mol
        self.verbose = bse.verbose
        self.stdout = bse.stdout
        self.max_memory = getattr(bse.mf, 'max_memory', self.mol.max_memory)

        self.nocc = int(bse.nocc[0])
        self.mo_energy = np.asarray(bse.mo_energy[0])
        self.mo_coeff = np.asarray(bse.mo_coeff[0])
        self.nmo = self.mo_energy.size
        self.Lpq = np.asarray(bse.Lpq[0])
        if np.count_nonzero(self.mo_energy == 0.0) > 0:
            raise RuntimeError(
                'Some quasiparticle energies are exactly zero: GW was probably restricted to gw.orbs. '
                'Apply pyscf.gw.qp_shift.scissor_shift(gw, inplace=True) before building BSE, '
                'or pass the GW object (scissor=True) instead.')
        if self.Lpq.shape[1] != self.nmo or self.mo_coeff.shape[1] != self.nmo:
            raise NotImplementedError('Frozen orbitals are not supported in BSE quadratic response.')

        # options
        self.screening = screening
        self.solver = 'iterative'
        self.conv_tol = 1e-9
        self.max_cycle = 300
        self.gauge_origin = (0.0, 0.0, 0.0)

        # intermediates
        self._Lbar = None
        self._apb = None
        self._amb = None

        # results
        self.alpha = None
        self.beta = None
        self.exci = None
        self.X_vec = None
        self.Y_vec = None
        self.tpa_tensor = None
        self.tpa_strength = None

    # ------------------------------------------------------------------ setup

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        o = self.nocc
        v = self.nmo - o
        log.info('')
        log.info('******** %s ********', self.__class__)
        log.info('nmo = %d  nocc = %d  nvir = %d  naux = %d', self.nmo, o, v, self.Lpq.shape[0])
        log.info('screened kernel (BSE) = %s', self.screening)
        log.info('linear solver = %s', self.solver)
        if self.solver == 'iterative':
            log.info('conv_tol = %g  max_cycle = %d', self.conv_tol, self.max_cycle)
        log.info('BSE hyperkernel neglected (static screening)')
        return self

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
        return self

    def _check_built(self):
        if self._Lbar is None:
            self.build()

    # ---------------------------------------------------------- kernel pieces

    def _orb_diff(self):
        o = self.nocc
        e = self.mo_energy
        return (e[None, o:] - e[:o, None]).ravel()

    def apb_amb_prod(self, zp, zm):
        '''Return (A+B) zp and (A-B) zm for vectors in the (occ, vir) flattened space.'''
        self._check_built()
        o = self.nocc
        tri = np.vstack([zp.ravel(), zm.ravel()])
        apb, amb, _ = _bse_contraction(
            's', [o], self.mo_energy[None], self._Lia, self._Laa,
            self._Lii_bar, self._Lia_bar, tri, TDA=False)
        return apb[0], amb[1]

    def get_dense_apb_amb(self):
        '''Dense A+B and A-B matrices (dimension nocc*nvir).'''
        if self._apb is None:
            self._check_built()
            o = self.nocc
            dim = o * (self.nmo - o)
            eye = np.eye(dim)
            apb, amb, _ = _bse_contraction(
                's', [o], self.mo_energy[None], self._Lia, self._Laa,
                self._Lii_bar, self._Lia_bar, eye, TDA=False)
            # symmetrize away round-off
            self._apb = (apb + apb.T) * 0.5
            self._amb = (amb + amb.T) * 0.5
        return self._apb, self._amb

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

    # --------------------------------------------------------- linear solver

    def solve(self, omega, r_ai, r_ia):
        '''Solve the coupled-perturbed BSE equations

            (A - w) X + B Y = -r_ai
            B X + (A + w) Y = -r_ia

        for one or several right-hand sides given as (n, nocc, nvir) arrays
        (r_ai[n, i, a] is the ai element, r_ia[n, i, a] the ia element).

        Returns X, Y with the same shape (X[n, i, a] = D1_ai, Y[n, i, a] = D1_ia).
        '''
        r_ai = np.asarray(r_ai, dtype=float)
        r_ia = np.asarray(r_ia, dtype=float)
        single = r_ai.ndim == 2
        if single:
            r_ai = r_ai[None]
            r_ia = r_ia[None]
        nrhs, o, v = r_ai.shape
        dim = o * v
        bp = -(r_ai + r_ia).reshape(nrhs, dim)
        bm = -(r_ai - r_ia).reshape(nrhs, dim)

        if self.solver == 'direct':
            zp, zm = self._solve_direct(omega, bp, bm)
        elif self.solver == 'iterative':
            zp, zm = self._solve_iterative(omega, bp, bm)
        else:
            raise ValueError('Unknown solver %s' % self.solver)

        X = ((zp + zm) * 0.5).reshape(nrhs, o, v)
        Y = ((zp - zm) * 0.5).reshape(nrhs, o, v)
        if single:
            return X[0], Y[0]
        return X, Y

    def _solve_direct(self, omega, bp, bm):
        apb, amb = self.get_dense_apb_amb()
        dim = apb.shape[0]
        eye = np.eye(dim)
        mat = np.block([[apb, -omega * eye], [-omega * eye, amb]])
        sol = scipy.linalg.solve(mat, np.hstack([bp, bm]).T, assume_a='sym')
        return sol[:dim].T, sol[dim:].T

    def _solve_iterative(self, omega, bp, bm):
        log = logger.new_logger(self)
        nrhs, dim = bp.shape
        d = self._orb_diff()
        denom = d * d - omega * omega
        if np.any(np.abs(denom) < 1e-12):
            raise ValueError('Frequency %g coincides with an orbital-energy difference.' % omega)

        def precond(x):
            xp, xm = x[:dim], x[dim:]
            return np.concatenate([(d * xp + omega * xm) / denom, (omega * xp + d * xm) / denom])

        def matvec(x):
            xp, xm = x[:dim], x[dim:]
            ap, am = self.apb_amb_prod(xp, xm)
            return np.concatenate([ap - omega * xm, am - omega * xp])

        op = spla.LinearOperator((2 * dim, 2 * dim), matvec=matvec, dtype=float)
        pc = spla.LinearOperator((2 * dim, 2 * dim), matvec=precond, dtype=float)

        zp = np.empty_like(bp)
        zm = np.empty_like(bm)
        for n in range(nrhs):
            b = np.concatenate([bp[n], bm[n]])
            bnorm = np.linalg.norm(b)
            if bnorm < 1e-14:
                zp[n] = 0.0
                zm[n] = 0.0
                continue
            x0 = precond(b)
            x, info = spla.gmres(op, b, x0=x0, rtol=self.conv_tol, atol=0.0,
                                 restart=min(60, 2 * dim), maxiter=self.max_cycle, M=pc)
            res = np.linalg.norm(matvec(x) - b) / bnorm
            if info != 0 or res > max(10 * self.conv_tol, 1e-6):
                log.warn('CP-BSE solver not converged at w = %g (rhs %d): relative residual %.2e',
                         omega, n, res)
            else:
                log.debug('CP-BSE rhs %d at w = %g converged, relative residual %.2e', n, omega, res)
            zp[n] = x[:dim]
            zm[n] = x[dim:]
        return zp, zm

    # ----------------------------------------------------- response building

    def get_dipole_mo(self):
        '''Dipole (position) integrals r_x, r_y, r_z in the MO basis, shape (3, nmo, nmo).'''
        with self.mol.with_common_orig(self.gauge_origin):
            ao_dip = self.mol.intor_symmetric('int1e_r', comp=3)
        c = self.mo_coeff
        return einsum('up,xuv,vq->xpq', c, ao_dip, c)

    def first_order(self, ops, omega):
        '''First-order response to perturbation operators.

        Parameters
        ----------
        ops : (n, nmo, nmo) array
            Symmetric perturbation operators in the MO basis.
        omega : float
            Frequency.

        Returns
        -------
        resp : dict with keys
            'X', 'Y' : (n, nocc, nvir) response amplitudes
            'D1' : (n, nmo, nmo) first-order (per-spin) density
            'F1' : (n, nmo, nmo) first-order effective Fock matrix V + G[D1]
            'omega' : the frequency
        '''
        ops = np.asarray(ops)
        o = self.nocc
        r_ai = ops[:, o:, :o].transpose(0, 2, 1)
        r_ia = ops[:, :o, o:]
        X, Y = self.solve(omega, r_ai, r_ia)
        D1 = self._ovvo_to_full(X, Y)
        F1 = ops + self.get_kernel(D1)
        return {'X': X, 'Y': Y, 'D1': D1, 'F1': F1, 'omega': omega}

    def _ovvo_to_full(self, X, Y):
        o = self.nocc
        D = np.zeros((X.shape[0], self.nmo, self.nmo))
        D[:, o:, :o] = X.transpose(0, 2, 1)
        D[:, :o, o:] = Y
        return D

    def second_order_rhs(self, resp_b, resp_c, ib, ic):
        '''Diagonal blocks K and right-hand side (r_ai, r_ia) of the second-order equations
        for perturbations resp_b[ib] at w1 and resp_c[ic] at w2 (Eqs. 25-28, hyperkernel = 0).
        '''
        o = self.nocc
        F1b, D1b = resp_b['F1'][ib], resp_b['D1'][ib]
        F1c, D1c = resp_c['F1'][ic], resp_c['D1'][ic]

        comm = F1b @ D1c - D1c @ F1b + F1c @ D1b - D1b @ F1c
        S = D1b @ D1c + D1c @ D1b
        K = np.zeros_like(S)
        K[:o, :o] = -S[:o, :o]
        K[o:, o:] = S[o:, o:]
        GK = self.get_kernel(K)

        r_ai = (comm[o:, :o] + GK[o:, :o]).T
        r_ia = -comm[:o, o:] + GK[:o, o:]
        return K, r_ai, r_ia

    def second_order_density(self, resp_b, resp_c, ib=0, ic=0):
        '''Explicit second-order (per-spin) density matrix D2^{BC}(w1, w2).'''
        K, r_ai, r_ia = self.second_order_rhs(resp_b, resp_c, ib, ic)
        omega = resp_b['omega'] + resp_c['omega']
        X2, Y2 = self.solve(omega, r_ai, r_ia)
        return K + self._ovvo_to_full(X2[None], Y2[None])[0]

    def quadratic_response(self, ops_a, ops_b, ops_c, omega1, omega2):
        '''Quadratic response function <<A; B, C>>_{w1, w2} for all operator combinations.

        Parameters
        ----------
        ops_a, ops_b, ops_c : (na, nmo, nmo), (nb, ...), (nc, ...) arrays
            Symmetric operators in the MO basis.
        omega1, omega2 : float
            Frequencies of B and C.  A is probed at -(omega1 + omega2).

        Returns
        -------
        qr : (na, nb, nc) array
        '''
        ops_a = np.asarray(ops_a)
        ops_b = np.asarray(ops_b)
        ops_c = np.asarray(ops_c)
        omega = omega1 + omega2
        resp_b = self.first_order(ops_b, omega1)
        resp_c = resp_b if (ops_c is ops_b and omega2 == omega1) else self.first_order(ops_c, omega2)
        resp_a = self.first_order(ops_a, omega)
        return self._contract_qr(ops_a, resp_a, resp_b, resp_c)

    def _contract_qr(self, ops_a, resp_a, resp_b, resp_c):
        na = ops_a.shape[0]
        nb = resp_b['X'].shape[0]
        nc = resp_c['X'].shape[0]
        o = self.nocc
        same = resp_b is resp_c
        qr = np.zeros((na, nb, nc))
        for ib in range(nb):
            for ic in range(nc):
                if same and ic < ib:
                    qr[:, ib, ic] = qr[:, ic, ib]
                    continue
                K, r_ai, r_ia = self.second_order_rhs(resp_b, resp_c, ib, ic)
                trAK = einsum('xpq,pq->x', ops_a[:, :o, :o], K[:o, :o]) \
                    + einsum('xpq,pq->x', ops_a[:, o:, o:], K[o:, o:])
                zr = einsum('xia,ia->x', resp_a['X'], r_ai) + einsum('xia,ia->x', resp_a['Y'], r_ia)
                qr[:, ib, ic] = 2.0 * (trAK + zr)
        return qr

    # ------------------------------------------------------------ properties

    def polarizability(self, omega=0.0):
        '''Dipole polarizability alpha(-w; w) in a.u.'''
        self._check_built()
        r = self.get_dipole_mo()
        resp = self.first_order(r, omega)
        o = self.nocc
        alpha = -2.0 * einsum('xia,yia->xy', r[:, :o, o:], resp['X'] + resp['Y'])
        self.alpha = alpha
        return alpha

    def hyperpolarizability(self, omega1=0.0, omega2=0.0):
        '''First hyperpolarizability beta_xyz(-w1-w2; w1, w2) in a.u. (Taylor convention).

        Special cases:
            static:   omega1 = omega2 = 0
            SHG:      omega1 = omega2 = w           -> beta(-2w; w, w)
            EOPE:     omega1 = w, omega2 = 0        -> beta(-w; w, 0)
            OR:       omega1 = w, omega2 = -w       -> beta(0; w, -w)
        '''
        cput0 = (logger.process_clock(), logger.perf_counter())
        self._check_built()
        self.dump_flags()
        r = self.get_dipole_mo()
        beta = -self.quadratic_response(r, r, r, omega1, omega2)
        self.beta = beta
        logger.timer(self, 'BSE hyperpolarizability', *cput0)
        logger.note(self, 'beta(-%.6f; %.6f, %.6f) (a.u.):', omega1 + omega2, omega1, omega2)
        if self.verbose >= logger.NOTE:
            for x, lab in enumerate('xyz'):
                logger.note(self, '  %s  %s', lab, np.array2string(beta[x], precision=5, suppress_small=True))
        return beta

    def get_excited_states(self, nstates):
        '''Lowest singlet BSE excited states consistent with this object's kernel.

        Uses the dense A+B/A-B matrices for small problems and otherwise the Davidson solver of
        pyscf.gw.bse (screened kernel only).  Eigenvectors are normalized to X.X - Y.Y = 1.
        '''
        o = self.nocc
        v = self.nmo - o
        dim = o * v
        nstates = min(nstates, dim)
        if self.solver == 'direct' or dim <= 3000:
            apb, amb = self.get_dense_apb_amb()
            w2, xpy = scipy.linalg.eigh(apb, amb, type=3, subset_by_index=[0, nstates - 1])
            if np.any(w2 <= 0):
                raise RuntimeError('BSE instability: non-positive squared excitation energy.')
            exci = np.sqrt(w2)
            xmy = (apb @ xpy) / exci[None, :]
            X = ((xpy + xmy) * 0.5).T
            Y = ((xpy - xmy) * 0.5).T
        else:
            if not self.screening:
                raise NotImplementedError('Use solver="direct" for unscreened excited states.')
            bse = self._bse
            bse.nroot = nstates
            bse.max_vec = max(bse.max_vec, 12 * nstates)
            exci, X_vec, Y_vec = bse.kernel('s')
            X = X_vec[0].reshape(-1, dim)
            Y = Y_vec[0].reshape(-1, dim)
        norm = np.einsum('ni,ni->n', X, X) - np.einsum('ni,ni->n', Y, Y)
        X = X / np.sqrt(norm)[:, None]
        Y = Y / np.sqrt(norm)[:, None]
        self.exci = np.asarray(exci)
        self.X_vec = X.reshape(-1, o, v)
        self.Y_vec = Y.reshape(-1, o, v)
        return self.exci, self.X_vec, self.Y_vec

    def two_photon_absorption(self, nstates=5, states=None):
        '''Degenerate two-photon absorption (each photon w = Omega_N / 2).

        Parameters
        ----------
        nstates : int
            Number of lowest singlet states to compute.
        states : list of int, optional
            Subset of state indices (0-based) to evaluate.

        Returns
        -------
        exci : (n,) excitation energies
        tensor : (n, 3, 3) two-photon transition tensors T^N (a.u.)
        delta : dict with rotationally averaged strengths (a.u.) for 'linear' and
            'circular' polarization of identical photons.
        '''
        cput0 = (logger.process_clock(), logger.perf_counter())
        self._check_built()
        self.dump_flags()
        if self.exci is None or len(self.exci) < nstates:
            self.get_excited_states(nstates)
        if states is None:
            states = range(min(nstates, len(self.exci)))
        states = list(states)

        r = self.get_dipole_mo()
        tensors = np.zeros((len(states), 3, 3))
        for k, n in enumerate(states):
            w = self.exci[n] * 0.5
            resp = self.first_order(r, w)
            tensors[k] = self.tpa_transition_tensor(n, resp, resp)

        delta_lin, delta_circ = _tpa_strength(tensors)
        self.tpa_tensor = tensors
        self.tpa_strength = {'linear': delta_lin, 'circular': delta_circ}

        log = logger.new_logger(self)
        log.note('Two-photon absorption (static-screened BSE, hyperkernel neglected)')
        log.note('  state   Omega (eV)    delta_lin (a.u.)   delta_circ (a.u.)')
        for k, n in enumerate(states):
            log.note('  %5d  %10.4f  %16.6e  %16.6e', n + 1, self.exci[n] * 27.211386245988,
                     delta_lin[k], delta_circ[k])
        logger.timer(self, 'BSE two-photon absorption', *cput0)
        return self.exci[states], tensors, self.tpa_strength

    def tpa_transition_tensor(self, n, resp_b, resp_c):
        '''Two-photon transition tensor T^N_BC = sqrt(2) z_N . r^BC(w1, w2), w1 + w2 = Omega_N.'''
        nb = resp_b['X'].shape[0]
        nc = resp_c['X'].shape[0]
        Xn = self.X_vec[n]
        Yn = self.Y_vec[n]
        T = np.zeros((nb, nc))
        for ib in range(nb):
            for ic in range(nc):
                _, r_ai, r_ia = self.second_order_rhs(resp_b, resp_c, ib, ic)
                T[ib, ic] = np.sqrt(2.0) * (np.vdot(Xn, r_ai) + np.vdot(Yn, r_ia))
        return T

    def transition_dipole(self):
        '''Ground-to-excited transition dipoles <0|r|N> (spin-summed), shape (nstates, 3).'''
        r = self.get_dipole_mo()
        o = self.nocc
        return np.sqrt(2.0) * einsum('xia,nia->nx', r[:, :o, o:], self.X_vec + self.Y_vec)

    kernel = hyperpolarizability


def beta_parallel(beta, direction):
    '''beta_|| = 1/5 sum_j u_i (beta_ijj + beta_jij + beta_jji) for unit vector u (e.g. dipole).'''
    u = np.asarray(direction, dtype=float)
    u = u / np.linalg.norm(u)
    vec = np.einsum('ijj->i', beta) + np.einsum('jij->i', beta) + np.einsum('jji->i', beta)
    return 0.2 * np.dot(u, vec)


def _tpa_strength(tensors):
    '''Rotationally averaged two-photon strengths, delta = (F dF + G dG + H dH) / 30.'''
    dF = np.einsum('nii,njj->n', tensors, tensors)
    dG = np.einsum('nij,nij->n', tensors, tensors)
    dH = np.einsum('nij,nji->n', tensors, tensors)
    delta_lin = (2 * dF + 2 * dG + 2 * dH) / 30.0
    delta_circ = (-2 * dF + 3 * dG + 3 * dH) / 30.0
    return delta_lin, delta_circ


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
