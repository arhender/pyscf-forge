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
Shared machinery for linear, quadratic and cubic response of restricted, closed-shell,
real-orbital time-dependent mean-field-like theories (TDHF, adiabatic TDDFT, static-screened BSE).

All such theories share the density-matrix equation of motion i dD/dt = [F, D] with
idempotency D^2 = D (per-spin density, D0 = occupied projector) and

    F[D] = F0 + V(t) + G[D - D0] + 1/2 g[D - D0, D - D0] + 1/6 h[D - D0, D - D0, D - D0],

where F0 = diag(eps) (Kohn-Sham, Hartree-Fock or GW quasiparticle energies) and G is the
method-specific kernel.  Subclasses provide G (get_kernel) and, optionally, the hyperkernel g
(_hyperkernel, e.g. the XC hyperkernel) and the third-order kernel h (_third_kernel, e.g. the
fourth functional derivative of E_xc), faster matrix-vector products (_lr_matvec), dense
matrices (get_dense_apb_amb) and an excited-state solver for large problems
(_excited_states_large).

First order (operator V at frequency w):
    (A - w) X + B Y = -V_ai,      B X + (A + w) Y = -V_ia,
    D1_ai = X_ia, D1_ia = Y_ia, F1 = V + G[D1]
Second order (B at w1, C at w2, w = w1 + w2):
    K_ij = -sum_a (Y^B_ia X^C_ja + Y^C_ia X^B_ja),   K_ab = sum_i (X^B_ia Y^C_ib + X^C_ia Y^B_ib)
    Comm = [F1^B, D1^C] + [F1^C, D1^B]
    r_ai = Comm_ai + G2_ai,   r_ia = -Comm_ia + G2_ia,   G2 = G[K] + g[D1^B, D1^C]
    D2^BC = K + (X^BC, Y^BC) from the linear equations at w1 + w2,
    F2^BC = G[D2^BC] + g[D1^B, D1^C]                       (no V at second order)
Quadratic response function (2n+1 form):
    <<A; B, C>>_{w1,w2} = 2 [ Tr(A K) + z_A(w1 + w2) . r^BC ]
Two-photon transition tensor (single residue):
    T^N_BC = sqrt(2) z_N . r^BC(w1, w2),  w1 + w2 = Omega_N,  X_N.X_N - Y_N.Y_N = 1.

Third order (B at w1, C at w2, D at w3, ws = w1 + w2 + w3; the sums run over the three
pairings X|YZ = B|CD, C|BD, D|BC):
    S3 = sum (D1^X D2^YZ + D2^YZ D1^X),   K3_oo = -S3_oo,   K3_vv = S3_vv   (idempotency;
         the ov block of S3 cancels in the sum)
    Comm3 = sum ([F1^X, D2^YZ] + [F2^YZ, D1^X])
    G3 = G[K3] + sum g[D1^X, D2^YZ] + h[D1^B, D1^C, D1^D]
    r3_ai = Comm3_ai + G3_ai,   r3_ia = -Comm3_ia + G3_ia
Cubic response function (2n+2 form; second-order but no third-order equations are solved):
    <<A; B, C, D>>_{w1,w2,w3} = 2 [ Tr(A K3) + z_A(ws) . r3^BCD ]
Three-photon transition tensor (single residue; K3 has no pole at ws = Omega_N):
    T^N_BCD = sqrt(2) z_N . r3^BCD(w1, w2, w3),  w1 + w2 + w3 = Omega_N.

Conventions: H' = F . r, Taylor convention
mu(F) = mu0 + alpha F + 1/2 beta F F + 1/6 gamma F F F, i.e.
alpha_xy = -<<r_x; r_y>>, beta_xyz(-w1-w2; w1, w2) = -<<r_x; r_y, r_z>>,
gamma_xyzw(-ws; w1, w2, w3) = -<<r_x; r_y, r_z, r_w>>.

References for cubic response: B. Jansik, P. Salek, D. Jonsson, O. Vahtras, H. Agren,
J. Chem. Phys. 122, 054107 (2005) (TDDFT cubic response); A. J. Thorvaldsen et al.,
J. Chem. Phys. 129, 214108 (2008) (density-matrix formulation, 2n+1/2n+2 rules);
E. Cronstrand, P. Norman, Y. Luo, H. Agren, J. Chem. Phys. 121, 2020 (2004) (three-photon
absorption from the cubic response residue).
See pyscf.tdscf.quadratic_response (TDHF/TDDFT) and pyscf.gw.bse_qr (BSE) for the other references.
'''

import itertools

import numpy as np
import scipy.linalg
import scipy.sparse.linalg as spla

from pyscf import lib
from pyscf.lib import logger

einsum = lib.einsum

HARTREE2EV = 27.211386245988


class QuadraticResponseBase(lib.StreamObject):
    '''Method-independent linear/quadratic response driver.

    Attributes:
        solver : str
            'iterative' (GMRES with a diagonal preconditioner, default) or 'direct'
            (dense A+B / A-B matrices; small systems only).
        conv_tol : float
            Relative residual tolerance of the iterative linear solver.
        max_cycle : int
            Maximum number of GMRES iterations per right-hand side.
        gauge_origin : 3-tuple
            Origin of the dipole operator.  Results for neutral molecules do not depend on it.
        max_dense_dim : int
            Excited states are obtained from dense matrices up to this nocc*nvir dimension.
    '''

    _label = 'linear response'

    def __init__(self, mol, mo_energy, mo_coeff, nocc, verbose=None, stdout=None, max_memory=None):
        self.mol = mol
        self.verbose = mol.verbose if verbose is None else verbose
        self.stdout = mol.stdout if stdout is None else stdout
        self.max_memory = mol.max_memory if max_memory is None else max_memory

        self.mo_energy = np.asarray(mo_energy)
        self.mo_coeff = np.asarray(mo_coeff)
        self.nocc = int(nocc)
        self.nmo = self.mo_energy.size

        # options
        self.solver = 'iterative'
        self.conv_tol = 1e-9
        self.max_cycle = 300
        self.gauge_origin = (0.0, 0.0, 0.0)
        self.max_dense_dim = 3000

        # intermediates
        self._built = False
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
        self.tpa_frequencies = None
        self.gamma = None
        self.tpa3_tensor = None
        self.tpa3_strength = None
        self.tpa3_frequencies = None

    # ------------------------------------------------------------------ setup

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        o = self.nocc
        log.info('')
        log.info('******** %s ********', self.__class__)
        log.info('method = %s', self._label)
        self._dump_method_flags(log)
        log.info('nmo = %d  nocc = %d  nvir = %d', self.nmo, o, self.nmo - o)
        log.info('linear solver = %s', self.solver)
        if self.solver == 'iterative':
            log.info('conv_tol = %g  max_cycle = %d', self.conv_tol, self.max_cycle)
        return self

    def _dump_method_flags(self, log):
        pass

    def build(self):
        '''Precompute method-specific intermediates.  Subclasses call this at the end.'''
        self._built = True
        return self

    def _check_built(self):
        if not self._built:
            self.build()

    # ------------------------------------------------------- method interface

    def get_kernel(self, dm):
        '''Kernel G[D] in the MO basis for one (nmo, nmo) or several (n, nmo, nmo) per-spin,
        possibly non-symmetric density matrices.'''
        raise NotImplementedError

    def _hyperkernel(self, Db, Dc):
        '''Hyperkernel g[Db, Dc] (MO basis) for per-spin, possibly non-symmetric densities,
        or None if the method has none.'''
        return None

    def _third_kernel(self, Db, Dc, Dd):
        '''Third-order kernel h[Db, Dc, Dd] (MO basis), or None if the method has none.'''
        return None

    def _second_order_kernel(self, K, D1b, D1c):
        '''Kernel terms of the second-order Fock matrix: G[K] plus any hyperkernel contribution.'''
        G2 = self.get_kernel(K)
        g = self._hyperkernel(D1b, D1c)
        return G2 if g is None else G2 + g

    def _lr_matvec(self, omega, X, Y):
        '''Apply [[A - w, B], [B, A + w]] to (X, Y), each of shape (nocc, nvir).'''
        o = self.nocc
        G = self.get_kernel(self._ovvo_to_full(X[None], Y[None])[0])
        d = self._orb_diff().reshape(X.shape)
        return (d - omega) * X + G[o:, :o].T, (d + omega) * Y + G[:o, o:]

    def get_dense_apb_amb(self):
        '''Dense A+B and A-B matrices (dimension nocc*nvir), built from get_kernel.'''
        if self._apb is None:
            self._check_built()
            o = self.nocc
            v = self.nmo - o
            dim = o * v
            a1 = np.empty((dim, dim))
            b = np.empty((dim, dim))
            blksize = max(1, int(self.max_memory * 1e6 / 8 / (4 * self.nmo**2 + 2 * self.mol.nao**2)))
            for p0 in range(0, dim, blksize):
                p1 = min(dim, p0 + blksize)
                X = np.zeros((p1 - p0, dim))
                X[np.arange(p1 - p0), np.arange(p0, p1)] = 1.0
                X = X.reshape(-1, o, v)
                G = self.get_kernel(self._ovvo_to_full(X, np.zeros_like(X)))
                a1[:, p0:p1] = G[:, o:, :o].transpose(0, 2, 1).reshape(p1 - p0, dim).T
                b[:, p0:p1] = G[:, :o, o:].reshape(p1 - p0, dim).T
            a1[np.diag_indices(dim)] += self._orb_diff()     # a1 = A
            a1 += b                                          # A + B
            b *= -2.0
            b += a1                                          # A - B
            self._set_dense(a1, b)
        return self._apb, self._amb

    def _set_dense(self, apb, amb):
        '''Store the symmetrized dense A+B and A-B.  apb and amb are overwritten.'''
        self._apb = lib.transpose_sum(apb, inplace=True)
        self._apb *= 0.5
        self._amb = lib.transpose_sum(amb, inplace=True)
        self._amb *= 0.5

    def _excited_states_large(self, nstates):
        '''Iterative excited-state solver for problems above max_dense_dim.
        Returns exci and X, Y of shape (nstates, nocc*nvir).'''
        raise NotImplementedError('Use a smaller system or solver="direct".')

    # ---------------------------------------------------------------- helpers

    def _orb_diff(self):
        o = self.nocc
        e = self.mo_energy
        return (e[None, o:] - e[:o, None]).ravel()

    def _mo2ao(self, dm):
        c = self.mo_coeff
        return einsum('pi,nij,qj->npq', c, dm, c)

    def _ao2mo(self, v):
        c = self.mo_coeff
        return einsum('pi,npq,qj->nij', c, v, c)

    def _ovvo_to_full(self, X, Y):
        o = self.nocc
        D = np.zeros((X.shape[0], self.nmo, self.nmo))
        D[:, o:, :o] = X.transpose(0, 2, 1)
        D[:, :o, o:] = Y
        return D

    def get_dipole_mo(self):
        '''Dipole (position) integrals r_x, r_y, r_z in the MO basis, shape (3, nmo, nmo).'''
        with self.mol.with_common_orig(self.gauge_origin):
            ao_dip = self.mol.intor_symmetric('int1e_r', comp=3)
        return self._ao2mo(ao_dip)

    # --------------------------------------------------------- linear solver

    def solve(self, omega, r_ai, r_ia):
        '''Solve the coupled-perturbed equations

            (A - w) X + B Y = -r_ai
            B X + (A + w) Y = -r_ia

        for one or several right-hand sides given as (n, nocc, nvir) arrays
        (r_ai[n, i, a] is the ai element, r_ia[n, i, a] the ia element).

        Returns X, Y with the same shape (X[n, i, a] = D1_ai, Y[n, i, a] = D1_ia).
        '''
        self._check_built()
        r_ai = np.asarray(r_ai, dtype=float)
        r_ia = np.asarray(r_ia, dtype=float)
        single = r_ai.ndim == 2
        if single:
            r_ai = r_ai[None]
            r_ia = r_ia[None]
        if self.solver == 'direct':
            X, Y = self._solve_direct(omega, r_ai, r_ia)
        elif self.solver == 'iterative':
            X, Y = self._solve_iterative(omega, r_ai, r_ia)
        else:
            raise ValueError('Unknown solver %s' % self.solver)
        if single:
            return X[0], Y[0]
        return X, Y

    def _solve_direct(self, omega, r_ai, r_ia):
        '''Dense solution of the response equations for zp = X + Y, zm = X - Y,

            (A+B) zp - w zm = bp,    (A-B) zm - w zp = bm,

        reduced to M zp = (A-B) bp + w bm and M^T zm = (A+B) bm + w bp with
        M = (A-B)(A+B) - w^2, which needs one dim x dim matrix and one LU factorization
        instead of the 2*dim block system.'''
        nrhs, o, v = r_ai.shape
        dim = o * v
        apb, amb = self.get_dense_apb_amb()
        bp = -(r_ai + r_ia).reshape(nrhs, dim).T
        bm = -(r_ai - r_ia).reshape(nrhs, dim).T
        M = amb @ apb
        M[np.diag_indices(dim)] -= omega * omega
        # factor M^T: it is the Fortran-ordered view of the C-ordered M, so LAPACK works in place
        lu = scipy.linalg.lu_factor(M.T, overwrite_a=True, check_finite=False)
        del M
        zp = scipy.linalg.lu_solve(lu, amb @ bp + omega * bm, trans=1, check_finite=False)
        zm = scipy.linalg.lu_solve(lu, apb @ bm + omega * bp, check_finite=False)
        zp, zm = zp.T, zm.T
        return ((zp + zm) * 0.5).reshape(nrhs, o, v), ((zp - zm) * 0.5).reshape(nrhs, o, v)

    def _solve_iterative(self, omega, r_ai, r_ia):
        log = logger.new_logger(self)
        nrhs, o, v = r_ai.shape
        dim = o * v
        d = self._orb_diff()
        if np.any(np.abs(d * d - omega * omega) < 1e-12):
            raise ValueError('Frequency %g coincides with an orbital-energy difference.' % omega)

        def matvec(x):
            ax, ay = self._lr_matvec(omega, x[:dim].reshape(o, v), x[dim:].reshape(o, v))
            return np.concatenate([ax.ravel(), ay.ravel()])

        def precond(x):
            return np.concatenate([x[:dim] / (d - omega), x[dim:] / (d + omega)])

        op = spla.LinearOperator((2 * dim, 2 * dim), matvec=matvec, dtype=float)
        pc = spla.LinearOperator((2 * dim, 2 * dim), matvec=precond, dtype=float)
        X = np.zeros((nrhs, o, v))
        Y = np.zeros((nrhs, o, v))
        for n in range(nrhs):
            b = -np.concatenate([r_ai[n].ravel(), r_ia[n].ravel()])
            bnorm = np.linalg.norm(b)
            if bnorm < 1e-14:
                continue
            x, info = spla.gmres(op, b, x0=precond(b), rtol=self.conv_tol, atol=0.0,
                                 restart=min(60, 2 * dim), maxiter=self.max_cycle, M=pc)
            res = np.linalg.norm(matvec(x) - b) / bnorm
            if info != 0 or res > max(10 * self.conv_tol, 1e-6):
                log.warn('%s linear solver not converged at w = %g (rhs %d): relative residual %.2e',
                         self._label, omega, n, res)
            else:
                log.debug('rhs %d at w = %g converged, relative residual %.2e', n, omega, res)
            X[n] = x[:dim].reshape(o, v)
            Y[n] = x[dim:].reshape(o, v)
        return X, Y

    # ----------------------------------------------------- response building

    def first_order(self, ops, omega):
        '''First-order response to symmetric MO-basis operators ops (n, nmo, nmo) at frequency omega.

        Returns a dict with 'X', 'Y' (n, nocc, nvir), 'D1' and 'F1' = V + G[D1] (n, nmo, nmo)
        and 'omega'.
        '''
        self._check_built()
        ops = np.asarray(ops)
        o = self.nocc
        X, Y = self.solve(omega, ops[:, o:, :o].transpose(0, 2, 1), ops[:, :o, o:])
        D1 = self._ovvo_to_full(X, Y)
        F1 = ops + self.get_kernel(D1)
        return {'X': X, 'Y': Y, 'D1': D1, 'F1': F1, 'omega': omega}

    def second_order_rhs(self, resp_b, resp_c, ib, ic):
        '''Diagonal blocks K and right-hand side (r_ai, r_ia) of the second-order equations
        for perturbations resp_b[ib] at w1 and resp_c[ic] at w2.'''
        o = self.nocc
        F1b, D1b = resp_b['F1'][ib], resp_b['D1'][ib]
        F1c, D1c = resp_c['F1'][ic], resp_c['D1'][ic]

        comm = F1b @ D1c - D1c @ F1b + F1c @ D1b - D1b @ F1c
        S = D1b @ D1c + D1c @ D1b
        K = np.zeros_like(S)
        K[:o, :o] = -S[:o, :o]
        K[o:, o:] = S[o:, o:]
        G2 = self._second_order_kernel(K, D1b, D1c)

        r_ai = (comm[o:, :o] + G2[o:, :o]).T
        r_ia = -comm[:o, o:] + G2[:o, o:]
        return K, r_ai, r_ia

    def second_order_density(self, resp_b, resp_c, ib=0, ic=0):
        '''Explicit second-order (per-spin) density matrix D2^{BC}(w1, w2).'''
        K, r_ai, r_ia = self.second_order_rhs(resp_b, resp_c, ib, ic)
        omega = resp_b['omega'] + resp_c['omega']
        X2, Y2 = self.solve(omega, r_ai, r_ia)
        return K + self._ovvo_to_full(X2[None], Y2[None])[0]

    def quadratic_response(self, ops_a, ops_b, ops_c, omega1, omega2):
        '''Quadratic response function <<A; B, C>>_{w1, w2} for all operator combinations.

        ops_* are symmetric MO-basis operators of shape (n*, nmo, nmo); A is probed at
        -(omega1 + omega2).  Returns an (na, nb, nc) array.
        '''
        ops_a = np.asarray(ops_a)
        ops_b = np.asarray(ops_b)
        ops_c = np.asarray(ops_c)
        resp_b = self.first_order(ops_b, omega1)
        resp_c = resp_b if (ops_c is ops_b and omega2 == omega1) else self.first_order(ops_c, omega2)
        resp_a = self.first_order(ops_a, omega1 + omega2)

        o = self.nocc
        same = resp_b is resp_c
        nb, nc = resp_b['X'].shape[0], resp_c['X'].shape[0]
        qr = np.zeros((ops_a.shape[0], nb, nc))
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

    def second_order(self, resp_b, resp_c, pairs):
        '''Second-order densities and Fock matrices for perturbation pairs.

        pairs is a list of component indices (ib, ic) of resp_b (at w1) and resp_c (at w2).
        All second-order linear equations at w1 + w2 are solved together.  Returns a list of
        (D2, F2) with D2 = K + (X^BC, Y^BC) and F2 = G[D2] + g[D1^B, D1^C] (no V term).
        '''
        if not pairs:
            return []
        rhs = [self.second_order_rhs(resp_b, resp_c, ib, ic) for ib, ic in pairs]
        omega = resp_b['omega'] + resp_c['omega']
        X2, Y2 = self.solve(omega, np.array([r[1] for r in rhs]), np.array([r[2] for r in rhs]))
        D2 = np.array([r[0] for r in rhs]) + self._ovvo_to_full(X2, Y2)
        F2 = self.get_kernel(D2)
        out = []
        for n, (ib, ic) in enumerate(pairs):
            g = self._hyperkernel(resp_b['D1'][ib], resp_c['D1'][ic])
            out.append((D2[n], F2[n] if g is None else F2[n] + g))
        return out

    def third_order_terms(self, resps):
        '''Third-order quantities for first-order responses resps = (resp_b, resp_c, resp_d).

        Returns a dict {(ib, ic, id): (K3, r3_ai, r3_ia)} over all component triples, where K3
        is the occupied-occupied/virtual-virtual block of the third-order density and r3 the
        right-hand side of the third-order linear equations at w1 + w2 + w3.  Identical
        response objects in resps are recognized, and permutation symmetry between them is
        used to skip redundant second- and third-order work.
        '''
        self._check_built()
        log = logger.new_logger(self)
        o = self.nocc
        resps = list(resps)
        if len(resps) != 3:
            raise ValueError('third_order_terms requires three first-order responses.')
        group = [next(j for j in range(k + 1) if resps[j] is resps[k]) for k in range(3)]
        ncomp = [r['X'].shape[0] for r in resps]
        slots = ((0, (1, 2)), (1, (0, 2)), (2, (0, 1)))

        def pkey(p, i, q, j):
            return tuple(sorted([(group[p], i), (group[q], j)]))

        # second-order densities / Fock matrices for the three pairings (cached by symmetry)
        d2 = {}
        for _, (p, q) in slots:
            todo = {}
            for i in range(ncomp[p]):
                for j in range(ncomp[q]):
                    key = pkey(p, i, q, j)
                    if key not in d2 and key not in todo:
                        todo[key] = (i, j)
            for key, res in zip(todo, self.second_order(resps[p], resps[q], list(todo.values()))):
                d2[key] = res
        log.debug('cubic response: %d second-order solutions', len(d2))

        # third-order commutators, K3 and kernel terms for symmetry-unique triples
        keys, K3s, partial = [], [], []
        canon = {}
        for idx in itertools.product(*[range(n) for n in ncomp]):
            key = tuple(sorted((group[k], idx[k]) for k in range(3)))
            canon[idx] = key
            if key in keys:
                continue
            comm = np.zeros((self.nmo, self.nmo))
            S = np.zeros((self.nmo, self.nmo))
            extra = np.zeros((self.nmo, self.nmo))
            for x, (p, q) in slots:
                D1 = resps[x]['D1'][idx[x]]
                F1 = resps[x]['F1'][idx[x]]
                D2, F2 = d2[pkey(p, idx[p], q, idx[q])]
                comm += F1 @ D2 - D2 @ F1 + F2 @ D1 - D1 @ F2
                S += D1 @ D2 + D2 @ D1
                g = self._hyperkernel(D1, D2)
                if g is not None:
                    extra += g
            h = self._third_kernel(*(resps[k]['D1'][idx[k]] for k in range(3)))
            if h is not None:
                extra += h
            K3 = np.zeros_like(S)
            K3[:o, :o] = -S[:o, :o]
            K3[o:, o:] = S[o:, o:]
            keys.append(key)
            K3s.append(K3)
            partial.append((comm, extra))
        log.debug('cubic response: %d third-order right-hand sides', len(keys))

        GK3 = self.get_kernel(np.array(K3s))
        terms = {}
        for key, K3, (comm, extra), G in zip(keys, K3s, partial, GK3):
            G3 = G + extra
            terms[key] = (K3, (comm[o:, :o] + G3[o:, :o]).T, -comm[:o, o:] + G3[:o, o:])
        return {idx: terms[key] for idx, key in canon.items()}

    def _first_order_set(self, ops_list, omegas):
        '''First-order responses for (operator, frequency) pairs, reusing identical ones.'''
        resps = []
        for k, (ops, w) in enumerate(zip(ops_list, omegas)):
            for j in range(k):
                if ops_list[j] is ops and omegas[j] == w:
                    resps.append(resps[j])
                    break
            else:
                resps.append(self.first_order(ops, w))
        return resps

    def cubic_response(self, ops_a, ops_b, ops_c, ops_d, omega1, omega2, omega3):
        '''Cubic response function <<A; B, C, D>>_{w1, w2, w3} for all operator combinations.

        ops_* are symmetric MO-basis operators of shape (n*, nmo, nmo); A is probed at
        -(omega1 + omega2 + omega3).  Returns an (na, nb, nc, nd) array.
        '''
        ops_a = np.asarray(ops_a)
        ops = [np.asarray(x) for x in (ops_b, ops_c, ops_d)]
        omegas = (omega1, omega2, omega3)
        omega_s = omega1 + omega2 + omega3
        resps = self._first_order_set(ops + [ops_a], omegas + (omega_s,))
        resp_a = resps.pop()
        terms = self.third_order_terms(resps)

        o = self.nocc
        cr = np.zeros((ops_a.shape[0],) + tuple(r['X'].shape[0] for r in resps))
        for idx, (K3, r_ai, r_ia) in terms.items():
            trAK = einsum('xpq,pq->x', ops_a[:, :o, :o], K3[:o, :o]) \
                + einsum('xpq,pq->x', ops_a[:, o:, o:], K3[o:, o:])
            zr = einsum('xia,ia->x', resp_a['X'], r_ai) + einsum('xia,ia->x', resp_a['Y'], r_ia)
            cr[(slice(None),) + idx] = 2.0 * (trAK + zr)
        return cr

    # ------------------------------------------------------------ properties

    def polarizability(self, omega=0.0):
        '''Dipole polarizability alpha(-w; w) in a.u.'''
        r = self.get_dipole_mo()
        resp = self.first_order(r, omega)
        o = self.nocc
        self.alpha = -2.0 * einsum('xia,yia->xy', r[:, :o, o:], resp['X'] + resp['Y'])
        return self.alpha

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
        self.beta = -self.quadratic_response(r, r, r, omega1, omega2)
        logger.timer(self, '%s hyperpolarizability' % self._label, *cput0)
        logger.note(self, 'beta(-%.6f; %.6f, %.6f) (a.u.):', omega1 + omega2, omega1, omega2)
        if self.verbose >= logger.NOTE:
            for x, lab in enumerate('xyz'):
                logger.note(self, '  %s  %s', lab, np.array2string(self.beta[x], precision=5, suppress_small=True))
        return self.beta

    def second_hyperpolarizability(self, omega1=0.0, omega2=0.0, omega3=0.0):
        '''Second hyperpolarizability gamma_xyzw(-ws; w1, w2, w3) in a.u. (Taylor convention),
        ws = w1 + w2 + w3.  Returns a (3, 3, 3, 3) array.

        Special cases:
            static:   (0, 0, 0)
            THG:      (w, w, w)       -> gamma(-3w; w, w, w)
            EFISHG:   (w, w, 0)       -> gamma(-2w; w, w, 0)
            DFWM:     (w, -w, w)      -> gamma(-w; w, -w, w)   (intensity-dependent refractive index)
            Kerr:     (w, 0, 0)       -> gamma(-w; w, 0, 0)
        '''
        cput0 = (logger.process_clock(), logger.perf_counter())
        self._check_built()
        self.dump_flags()
        r = self.get_dipole_mo()
        self.gamma = -self.cubic_response(r, r, r, r, omega1, omega2, omega3)
        logger.timer(self, '%s second hyperpolarizability' % self._label, *cput0)
        logger.note(self, 'gamma(-%.6f; %.6f, %.6f, %.6f) (a.u.): gamma_|| = %.6f',
                    omega1 + omega2 + omega3, omega1, omega2, omega3, gamma_parallel(self.gamma))
        if self.verbose >= logger.INFO:
            for x, lab in enumerate('xyz'):
                for y, lab2 in enumerate('xyz'):
                    logger.info(self, '  %s%s  %s', lab, lab2,
                                np.array2string(self.gamma[x, y], precision=4, suppress_small=True))
        return self.gamma

    def get_excited_states(self, nstates):
        '''Lowest singlet excited states consistent with this object's kernel,
        normalized to X.X - Y.Y = 1.'''
        o = self.nocc
        v = self.nmo - o
        dim = o * v
        nstates = min(nstates, dim)
        if self.solver == 'direct' or dim <= self.max_dense_dim:
            apb, amb = self.get_dense_apb_amb()
            w2, xpy = scipy.linalg.eigh(apb, amb, type=3, subset_by_index=[0, nstates - 1])
            if np.any(w2 <= 0):
                raise RuntimeError('Instability: non-positive squared excitation energy.')
            exci = np.sqrt(w2)
            xmy = (apb @ xpy) / exci[None, :]
            X = ((xpy + xmy) * 0.5).T
            Y = ((xpy - xmy) * 0.5).T
        else:
            exci, X, Y = self._excited_states_large(nstates)
        norm = np.einsum('ni,ni->n', X, X) - np.einsum('ni,ni->n', Y, Y)
        X = X / np.sqrt(norm)[:, None]
        Y = Y / np.sqrt(norm)[:, None]
        self.exci = np.asarray(exci)
        self.X_vec = X.reshape(-1, o, v)
        self.Y_vec = Y.reshape(-1, o, v)
        return self.exci, self.X_vec, self.Y_vec

    def tpa_transition_tensor(self, n, resp_b, resp_c):
        '''Two-photon transition tensor T^N_BC = sqrt(2) z_N . r^BC(w1, w2), w1 + w2 = Omega_N.'''
        nb, nc = resp_b['X'].shape[0], resp_c['X'].shape[0]
        T = np.zeros((nb, nc))
        for ib in range(nb):
            for ic in range(nc):
                if resp_b is resp_c and ic < ib:
                    T[ib, ic] = T[ic, ib]
                    continue
                _, r_ai, r_ia = self.second_order_rhs(resp_b, resp_c, ib, ic)
                T[ib, ic] = np.sqrt(2.0) * (np.vdot(self.X_vec[n], r_ai) + np.vdot(self.Y_vec[n], r_ia))
        return T

    def two_photon_absorption(self, nstates=5, states=None, frequencies=None):
        '''Two-photon absorption, degenerate (default) or non-degenerate.

        Parameters
        ----------
        nstates : int
            Number of lowest singlet states to compute.
        states : list of int, optional
            Subset of state indices (0-based) to evaluate.
        frequencies : optional
            Photon frequencies (w1, w2) in a.u.  The residue requires w1 + w2 = Omega_N.

            * None (default): degenerate 2PA, w1 = w2 = Omega_N / 2 for every state.
            * A single pair (w1, w2): used for every evaluated state.  One element may be
              None and is then set to Omega_N minus the other, e.g. (0.1, None) fixes one
              photon at 0.1 Eh for all states.  If both are numbers, they must sum to
              Omega_N for every evaluated state.
            * A sequence of pairs, one per evaluated state (in the order of `states`),
              each following the same rules.

        Returns
        -------
        exci : (n,) excitation energies
        tensor : (n, 3, 3) two-photon transition tensors T^N_BC(w1, w2) (a.u.); the first
            index belongs to the photon at w1, the second to the photon at w2.  Non-symmetric
            for non-degenerate photons, with T_BC(w1, w2) = T_CB(w2, w1).
        delta : dict with rotationally averaged strengths (a.u.) for 'linear' (parallel)
            and 'circular' (same sense, co-propagating) polarization.  The photon
            frequencies used are stored in self.tpa_frequencies, shape (n, 2).
        '''
        cput0 = (logger.process_clock(), logger.perf_counter())
        self._check_built()
        self.dump_flags()
        if self.exci is None or len(self.exci) < nstates:
            self.get_excited_states(nstates)
        if states is None:
            states = range(min(nstates, len(self.exci)))
        states = list(states)
        freqs = self._tpa_frequency_pairs(states, frequencies)
        degenerate = frequencies is None

        log = logger.new_logger(self)
        r = self.get_dipole_mo()
        tensors = np.zeros((len(states), 3, 3))
        for k, n in enumerate(states):
            w1, w2 = freqs[k]
            for w in (w1, w2):
                if np.any(np.abs(np.abs(w) - self.exci) < 1e-3):
                    log.warn('Photon frequency %.6f is (near-)resonant with an excited state; '
                             'the 2PA tensor of state %d may be unreliable.', w, n + 1)
            resp_b = self.first_order(r, w1)
            resp_c = resp_b if w1 == w2 else self.first_order(r, w2)
            tensors[k] = self.tpa_transition_tensor(n, resp_b, resp_c)

        delta_lin, delta_circ = _tpa_strength(tensors)
        self.tpa_tensor = tensors
        self.tpa_strength = {'linear': delta_lin, 'circular': delta_circ}
        self.tpa_frequencies = freqs

        log.note('%s two-photon absorption (%s quadratic response)',
                 'Degenerate' if degenerate else 'Non-degenerate', self._label)
        if degenerate:
            log.note('  state   Omega (eV)    delta_lin (a.u.)   delta_circ (a.u.)')
            for k, n in enumerate(states):
                log.note('  %5d  %10.4f  %16.6e  %16.6e', n + 1, self.exci[n] * HARTREE2EV,
                         delta_lin[k], delta_circ[k])
        else:
            log.note('  state   Omega (eV)   w1 (eV)    w2 (eV)    delta_lin (a.u.)   delta_circ (a.u.)')
            for k, n in enumerate(states):
                log.note('  %5d  %10.4f  %9.4f  %9.4f  %16.6e  %16.6e', n + 1,
                         self.exci[n] * HARTREE2EV, freqs[k, 0] * HARTREE2EV,
                         freqs[k, 1] * HARTREE2EV, delta_lin[k], delta_circ[k])
        logger.timer(self, '%s two-photon absorption' % self._label, *cput0)
        return self.exci[states], tensors, self.tpa_strength

    def three_photon_absorption(self, nstates=5, states=None, frequencies=None):
        '''Three-photon absorption from the single residue of the cubic response function.

        Parameters
        ----------
        nstates : int
            Number of lowest singlet states to compute.
        states : list of int, optional
            Subset of state indices (0-based) to evaluate.
        frequencies : optional
            Photon frequencies (w1, w2, w3) in a.u.  The residue requires w1 + w2 + w3 = Omega_N.

            * None (default): degenerate 3PA, w1 = w2 = w3 = Omega_N / 3 for every state.
            * A single triple, used for every evaluated state.  Entries given as None share
              the remainder Omega_N minus the given entries equally, e.g. (0.1, None, None).
              If all three are numbers, they must sum to Omega_N for every evaluated state.
            * A sequence of triples, one per evaluated state (in the order of `states`).

        Returns
        -------
        exci : (n,) excitation energies
        tensor : (n, 3, 3, 3) three-photon transition tensors T^N_BCD(w1, w2, w3) (a.u.); index
            k belongs to the photon at w_k.  Fully symmetric for degenerate photons.
        delta : dict with the rotationally averaged strength (a.u.) for 'linear' polarization
            (all three photons polarized along the same axis),
            delta = <|e_i e_j e_k T_ijk|^2>_orientations.  The photon frequencies used are
            stored in self.tpa3_frequencies, shape (n, 3).
        '''
        cput0 = (logger.process_clock(), logger.perf_counter())
        self._check_built()
        self.dump_flags()
        if self.exci is None or len(self.exci) < nstates:
            self.get_excited_states(nstates)
        if states is None:
            states = range(min(nstates, len(self.exci)))
        states = list(states)
        freqs = self._photon_frequencies(states, frequencies, 3)
        degenerate = frequencies is None

        log = logger.new_logger(self)
        r = self.get_dipole_mo()
        tensors = np.zeros((len(states), 3, 3, 3))
        for k, n in enumerate(states):
            ws = tuple(freqs[k])
            for i, j in ((0, 1), (0, 2), (1, 2)):
                if np.any(np.abs(np.abs(ws[i] + ws[j]) - self.exci) < 1e-3):
                    log.warn('Two-photon frequency %.6f is (near-)resonant with an excited state; '
                             'the 3PA tensor of state %d may be unreliable.', ws[i] + ws[j], n + 1)
            for w in ws:
                if np.any(np.abs(np.abs(w) - self.exci) < 1e-3):
                    log.warn('Photon frequency %.6f is (near-)resonant with an excited state; '
                             'the 3PA tensor of state %d may be unreliable.', w, n + 1)
            resps = self._first_order_set([r, r, r], ws)
            for idx, (_, r_ai, r_ia) in self.third_order_terms(resps).items():
                tensors[(k,) + idx] = np.sqrt(2.0) * (np.vdot(self.X_vec[n], r_ai)
                                                      + np.vdot(self.Y_vec[n], r_ia))

        delta_lin = _tpa3_strength(tensors)
        self.tpa3_tensor = tensors
        self.tpa3_strength = {'linear': delta_lin}
        self.tpa3_frequencies = freqs

        log.note('%s three-photon absorption (%s cubic response)',
                 'Degenerate' if degenerate else 'Non-degenerate', self._label)
        log.note('  state   Omega (eV)   w1 (eV)    w2 (eV)    w3 (eV)    delta_lin (a.u.)')
        for k, n in enumerate(states):
            log.note('  %5d  %10.4f  %9.4f  %9.4f  %9.4f  %16.6e', n + 1, self.exci[n] * HARTREE2EV,
                     *(freqs[k] * HARTREE2EV), delta_lin[k])
        logger.timer(self, '%s three-photon absorption' % self._label, *cput0)
        return self.exci[states], tensors, self.tpa3_strength

    def _tpa_frequency_pairs(self, states, frequencies, tol=1e-6):
        '''Resolve the `frequencies` argument of two_photon_absorption into an
        (len(states), 2) array of photon frequencies with w1 + w2 = Omega_N.'''
        return self._photon_frequencies(states, frequencies, 2, tol)

    def _photon_frequencies(self, states, frequencies, nphoton, tol=1e-6):
        '''Resolve a `frequencies` argument into an (len(states), nphoton) array of photon
        frequencies summing to Omega_N.  None entries share the remainder equally.'''
        omegas = np.asarray([self.exci[n] for n in states])
        if frequencies is None:
            return np.repeat(omegas[:, None] / nphoton, nphoton, axis=1)

        def is_tuple(p):
            return (isinstance(p, (tuple, list, np.ndarray)) and len(p) == nphoton
                    and all(x is None or np.ndim(x) == 0 for x in p))

        name = {2: 'pair', 3: 'triple'}.get(nphoton, '%d-tuple' % nphoton)
        if is_tuple(frequencies):
            tuples = [frequencies] * len(states)
        else:
            tuples = list(frequencies)
            if len(tuples) != len(states) or not all(is_tuple(p) for p in tuples):
                raise ValueError('frequencies must be None, a %s or a sequence of %d %ss, one '
                                 'per evaluated state.' % (name, len(states), name))

        out = np.empty((len(states), nphoton))
        for k, (n, ws) in enumerate(zip(states, tuples)):
            omega = omegas[k]
            free = [i for i, w in enumerate(ws) if w is None]
            given = sum(w for w in ws if w is not None)
            if free:
                ws = [(omega - given) / len(free) if w is None else w for w in ws]
            elif abs(given - omega) > tol:
                raise ValueError('Photon frequencies %s do not sum to the excitation energy %.8f '
                                 'of state %d (difference %.2e). Pass one of them as None to fix '
                                 'it from the others.'
                                 % (', '.join('%.8f' % w for w in ws), omega, n + 1, given - omega))
            else:
                ws = list(ws)
                ws[-1] = omega - sum(ws[:-1])   # remove the residual mismatch exactly
            out[k] = ws
        return out

    def transition_dipole(self):
        '''Ground-to-excited transition dipoles <0|r|N> (spin-summed), shape (nstates, 3).'''
        r = self.get_dipole_mo()
        o = self.nocc
        return np.sqrt(2.0) * einsum('xia,nia->nx', r[:, :o, o:], self.X_vec + self.Y_vec)

    def kernel(self, omega1=0.0, omega2=0.0):
        return self.hyperpolarizability(omega1, omega2)


def beta_parallel(beta, direction):
    '''beta_|| = 1/5 sum_j u_i (beta_ijj + beta_jij + beta_jji) for unit vector u (e.g. dipole).'''
    u = np.asarray(direction, dtype=float)
    u = u / np.linalg.norm(u)
    vec = np.einsum('ijj->i', beta) + np.einsum('jij->i', beta) + np.einsum('jji->i', beta)
    return 0.2 * np.dot(u, vec)


def gamma_parallel(gamma):
    '''Isotropic average gamma_|| = 1/15 sum_ij (gamma_iijj + gamma_ijij + gamma_ijji).'''
    return (np.einsum('iijj->', gamma) + np.einsum('ijij->', gamma) + np.einsum('ijji->', gamma)) / 15.0


def _perfect_matchings(items):
    if not items:
        yield []
        return
    first, rest = items[0], items[1:]
    for k in range(len(rest)):
        for m in _perfect_matchings(rest[:k] + rest[k + 1:]):
            yield [(first, rest[k])] + m


def _tpa3_strength(tensors):
    '''Rotationally averaged three-photon strength for linear polarization,
    delta = <(e_i e_j e_k T_ijk)^2> = 1/105 sum over the 15 pairings of the six indices of
    T_ijk T_lmn (valid for non-symmetric T; reduces to (3 F + 2 G) / 35 with
    F = sum_j (sum_i T_iij)^2, G = sum_ijk T_ijk^2 for a fully symmetric tensor).'''
    tensors = np.asarray(tensors)
    total = np.zeros(tensors.shape[0])
    for match in _perfect_matchings(list(range(6))):
        sub = [''] * 6
        for c, (u, v) in zip('ijk', match):
            sub[u] = sub[v] = c
        total += np.einsum('n%s,n%s->n' % (''.join(sub[:3]), ''.join(sub[3:])), tensors, tensors)
    return total / 105.0


def _tpa_strength(tensors):
    '''Rotationally averaged two-photon strengths, delta = (F dF + G dG + H dH) / 30.'''
    dF = np.einsum('nii,njj->n', tensors, tensors)
    dG = np.einsum('nij,nij->n', tensors, tensors)
    dH = np.einsum('nij,nji->n', tensors, tensors)
    delta_lin = (2 * dF + 2 * dG + 2 * dH) / 30.0
    delta_circ = (-2 * dF + 3 * dG + 3 * dH) / 30.0
    return delta_lin, delta_circ
