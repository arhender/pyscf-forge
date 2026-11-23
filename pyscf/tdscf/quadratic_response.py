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
Linear and quadratic response for restricted TDHF / TDDFT (adiabatic, closed shell, real orbitals).

Theory
------
S. M. Parker, D. Rappoport, F. Furche, J. Chem. Theory Comput. 14, 807 (2018), Sec. 1.

The response equations follow from the Kohn-Sham equation of motion i dD/dt = [F, D] and
idempotency D^2 = D (per-spin density, D0 = occupied projector), with

    F[D] = F0 + V(t) + G[D - D0] + 1/2 g_xc[D - D0, D - D0] + ...
    G[D]_pq   = 2 (pq|rs) D_rs - c_x (pr|sq) D_rs + f_xc[rho(2D)]_pq        (Parker Eq. 30)
    g_xc[B,C] = int phi_p phi_q (d^3 E_xc / d rho^3)(rho0) rho(2B) rho(2C)  (Parker Eqs. 31-32)

(range-separated exchange is handled by pyscf's response function).

First order (operator V at frequency w):
    (A - w) X + B Y = -V_ai,      B X + (A + w) Y = -V_ia,
    D1_ai = X_ia, D1_ia = Y_ia, F1 = V + G[D1]                               (Parker Eqs. 14, 29)

Second order (B at w1, C at w2, w = w1 + w2):
    K_ij = -sum_a (Y^B_ia X^C_ja + Y^C_ia X^B_ja),
    K_ab =  sum_i (X^B_ia Y^C_ib + X^C_ia Y^B_ib)                            (Parker Eq. 24)
    Comm = [F1^B, D1^C] + [F1^C, D1^B]
    r_ai = Comm_ai + G[K]_ai + g_xc[D1^B, D1^C]_ai,
    r_ia = -Comm_ia + G[K]_ia + g_xc[D1^B, D1^C]_ia                          (Parker Eq. 28)

The symmetric (A+B) and skew-symmetric (A-B) channels of Parker Eq. 28 are r_ai +/- r_ia;
G[K]_ai + G[K]_ia = 2 H+[K+] and G[K]_ai - G[K]_ia = 2 H-[K-] follow from the symmetry of G.

Quadratic response function (2n+1 form, Parker Eq. 33):
    <<A; B, C>>_{w1,w2} = 2 [ Tr(A K) + z_A(w1 + w2) . r^BC ]
Two-photon transition tensor (single residue, Parker Eqs. 34-35):
    T^N_BC = sqrt(2) z_N . r^BC(w1, w2),  w1 + w2 = Omega_N,  X_N.X_N - Y_N.Y_N = 1.

Conventions are those of pyscf.gw.bse_qr: H' = F . r, Taylor convention
mu(F) = mu0 + alpha F + 1/2 beta F F, so alpha_xy = -<<r_x; r_y>>, beta_xyz = -<<r_x; r_y, r_z>>.
'''

import numpy as np
import scipy.linalg
import scipy.sparse.linalg as spla

from pyscf import lib, scf
from pyscf.lib import logger

einsum = lib.einsum


class TDDFTQuadraticResponse(lib.StreamObject):
    '''Linear and quadratic response solver for restricted TDHF / adiabatic TDDFT.

    Attributes:
        hyperkernel : bool
            Include the exchange-correlation hyperkernel g_xc (third functional derivative).
            Required for consistency with finite-field derivatives.  Default True.
        solver : str
            'iterative' (GMRES on the coupled equations, default) or 'direct' (dense A, B from
            pyscf.tdscf.rhf.get_ab; small systems only).
        conv_tol : float
            Relative residual tolerance of the iterative linear solver.
        max_cycle : int
            Maximum number of GMRES iterations per right-hand side.
        gauge_origin : 3-tuple
            Origin of the dipole operator.
    '''

    def __init__(self, mf):
        if not isinstance(mf, scf.hf.RHF) or isinstance(mf, (scf.uhf.UHF, scf.rohf.ROHF)):
            raise NotImplementedError('TDDFT quadratic response is implemented for RHF/RKS references only.')
        if mf.mo_coeff is None:
            raise RuntimeError('The mean-field object must be converged before quadratic response.')
        self._scf = mf
        self.mol = mf.mol
        self.verbose = mf.verbose
        self.stdout = mf.stdout
        self.max_memory = mf.max_memory

        self.mo_energy = np.asarray(mf.mo_energy)
        self.mo_coeff = np.asarray(mf.mo_coeff)
        self.mo_occ = np.asarray(mf.mo_occ)
        self.nocc = int(np.count_nonzero(self.mo_occ > 0))
        self.nmo = self.mo_energy.size

        # options
        self.hyperkernel = True
        self.solver = 'iterative'
        self.conv_tol = 1e-9
        self.max_cycle = 300
        self.gauge_origin = (0.0, 0.0, 0.0)

        # intermediates
        self._vind = None
        self._xc = None
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

    @property
    def is_dft(self):
        return isinstance(self._scf, scf.hf.KohnShamDFT)

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        o = self.nocc
        log.info('')
        log.info('******** %s ********', self.__class__)
        log.info('reference = %s', self._scf.__class__.__name__)
        if self.is_dft:
            log.info('xc = %s', self._scf.xc)
            log.info('xc hyperkernel = %s', self.hyperkernel)
        log.info('nmo = %d  nocc = %d  nvir = %d', self.nmo, o, self.nmo - o)
        log.info('linear solver = %s', self.solver)
        if self.solver == 'iterative':
            log.info('conv_tol = %g  max_cycle = %d', self.conv_tol, self.max_cycle)
        return self

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
        return self

    def _check_built(self):
        if self._vind is None:
            self.build()

    # ---------------------------------------------------------- kernel pieces

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
        if self._xc is None:
            return np.zeros((self.nmo, self.nmo))
        xc = self._xc
        rho_c = self._rho_on_grid(2.0 * self._mo2ao(np.asarray(dm_c)[None])[0])
        kxc = xc['kxc']
        nvar = rho_c.shape[0]
        kxc = kxc.reshape(nvar, nvar, nvar, -1)
        fxc_eff = einsum('xyzg,zg->xyg', kxc, rho_c)
        dmb_ao = 2.0 * self._mo2ao(np.asarray(dm_b)[None])[0]
        v = xc['ni'].nr_rks_fxc(self.mol, xc['grids'], self._scf.xc, None, dmb_ao, 0, 0,
                                xc['rho0'], xc['vxc'], fxc_eff, max_memory=self.max_memory)
        return self._ao2mo(np.asarray(v).reshape(1, *v.shape[-2:]))[0]

    def get_dense_apb_amb(self):
        '''Dense A+B and A-B matrices (dimension nocc*nvir) built from get_kernel.

        Built from the same response function as the iterative solver, so they are consistent
        with the reference's integral scheme (e.g. density fitting) and grids.  This is not
        guaranteed by pyscf.tdscf.rhf.get_ab, which uses exact integrals for DF references.
        '''
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
            a = a1 + np.diag(self._orb_diff())
            apb = a + b
            amb = a - b
            self._apb = (apb + apb.T) * 0.5
            self._amb = (amb + amb.T) * 0.5
        return self._apb, self._amb

    # --------------------------------------------------------- linear solver

    def solve(self, omega, r_ai, r_ia):
        '''Solve (A - w) X + B Y = -r_ai,  B X + (A + w) Y = -r_ia  for (n, nocc, nvir) inputs.'''
        r_ai = np.asarray(r_ai, dtype=float)
        r_ia = np.asarray(r_ia, dtype=float)
        single = r_ai.ndim == 2
        if single:
            r_ai = r_ai[None]
            r_ia = r_ia[None]
        nrhs, o, v = r_ai.shape
        dim = o * v
        if self.solver == 'direct':
            apb, amb = self.get_dense_apb_amb()
            eye = np.eye(dim)
            mat = np.block([[apb, -omega * eye], [-omega * eye, amb]])
            bp = -(r_ai + r_ia).reshape(nrhs, dim)
            bm = -(r_ai - r_ia).reshape(nrhs, dim)
            sol = scipy.linalg.solve(mat, np.hstack([bp, bm]).T)
            zp, zm = sol[:dim].T, sol[dim:].T
            X = ((zp + zm) * 0.5).reshape(nrhs, o, v)
            Y = ((zp - zm) * 0.5).reshape(nrhs, o, v)
        elif self.solver == 'iterative':
            X, Y = self._solve_iterative(omega, r_ai, r_ia)
        else:
            raise ValueError('Unknown solver %s' % self.solver)
        if single:
            return X[0], Y[0]
        return X, Y

    def _lr_matvec(self, omega, X, Y):
        '''Apply [[A - w, B], [B, A + w]] to (X, Y); one response-function call per vector.'''
        o = self.nocc
        G = self.get_kernel(self._ovvo_to_full(X[None], Y[None])[0])
        d = self._orb_diff().reshape(X.shape)
        return (d - omega) * X + G[o:, :o].T, (d + omega) * Y + G[:o, o:]

    def _solve_iterative(self, omega, r_ai, r_ia):
        self._check_built()
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
                log.warn('CP-TDDFT solver not converged at w = %g (rhs %d): relative residual %.2e',
                         omega, n, res)
            X[n] = x[:dim].reshape(o, v)
            Y[n] = x[dim:].reshape(o, v)
        return X, Y

    # ----------------------------------------------------- response building

    def get_dipole_mo(self):
        '''Dipole (position) integrals r_x, r_y, r_z in the MO basis, shape (3, nmo, nmo).'''
        with self.mol.with_common_orig(self.gauge_origin):
            ao_dip = self.mol.intor_symmetric('int1e_r', comp=3)
        return self._ao2mo(ao_dip)

    def _ovvo_to_full(self, X, Y):
        o = self.nocc
        D = np.zeros((X.shape[0], self.nmo, self.nmo))
        D[:, o:, :o] = X.transpose(0, 2, 1)
        D[:, :o, o:] = Y
        return D

    def first_order(self, ops, omega):
        '''First-order response: dict with X, Y (n, nocc, nvir), D1 and F1 = V + G[D1] (n, nmo, nmo).'''
        self._check_built()
        ops = np.asarray(ops)
        o = self.nocc
        X, Y = self.solve(omega, ops[:, o:, :o].transpose(0, 2, 1), ops[:, :o, o:])
        D1 = self._ovvo_to_full(X, Y)
        F1 = ops + self.get_kernel(D1)
        return {'X': X, 'Y': Y, 'D1': D1, 'F1': F1, 'omega': omega}

    def second_order_rhs(self, resp_b, resp_c, ib, ic):
        '''K blocks and second-order right-hand side (r_ai, r_ia) (Parker Eqs. 24, 28).'''
        o = self.nocc
        F1b, D1b = resp_b['F1'][ib], resp_b['D1'][ib]
        F1c, D1c = resp_c['F1'][ic], resp_c['D1'][ic]

        comm = F1b @ D1c - D1c @ F1b + F1c @ D1b - D1b @ F1c
        S = D1b @ D1c + D1c @ D1b
        K = np.zeros_like(S)
        K[:o, :o] = -S[:o, :o]
        K[o:, o:] = S[o:, o:]
        GK = self.get_kernel(K)
        if self.hyperkernel:
            GK = GK + self.get_hyperkernel(D1b, D1c)

        r_ai = (comm[o:, :o] + GK[o:, :o]).T
        r_ia = -comm[:o, o:] + GK[:o, o:]
        return K, r_ai, r_ia

    def quadratic_response(self, ops_a, ops_b, ops_c, omega1, omega2):
        '''Quadratic response function <<A; B, C>>_{w1, w2}, shape (na, nb, nc).'''
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

    # ------------------------------------------------------------ properties

    def polarizability(self, omega=0.0):
        '''Dipole polarizability alpha(-w; w) in a.u.'''
        r = self.get_dipole_mo()
        resp = self.first_order(r, omega)
        o = self.nocc
        self.alpha = -2.0 * einsum('xia,yia->xy', r[:, :o, o:], resp['X'] + resp['Y'])
        return self.alpha

    def hyperpolarizability(self, omega1=0.0, omega2=0.0):
        '''First hyperpolarizability beta_xyz(-w1-w2; w1, w2) in a.u. (Taylor convention).'''
        cput0 = (logger.process_clock(), logger.perf_counter())
        self._check_built()
        self.dump_flags()
        r = self.get_dipole_mo()
        self.beta = -self.quadratic_response(r, r, r, omega1, omega2)
        logger.timer(self, 'TDDFT hyperpolarizability', *cput0)
        logger.note(self, 'beta(-%.6f; %.6f, %.6f) (a.u.):', omega1 + omega2, omega1, omega2)
        if self.verbose >= logger.NOTE:
            for x, lab in enumerate('xyz'):
                logger.note(self, '  %s  %s', lab, np.array2string(self.beta[x], precision=5, suppress_small=True))
        return self.beta

    def get_excited_states(self, nstates):
        '''Lowest singlet excited states, normalized to X.X - Y.Y = 1.'''
        o = self.nocc
        v = self.nmo - o
        dim = o * v
        nstates = min(nstates, dim)
        if self.solver == 'direct' or dim <= 3000:
            apb, amb = self.get_dense_apb_amb()
            w2, xpy = scipy.linalg.eigh(apb, amb, type=3, subset_by_index=[0, nstates - 1])
            if np.any(w2 <= 0):
                raise RuntimeError('Triplet/singlet instability: non-positive squared excitation energy.')
            exci = np.sqrt(w2)
            xmy = (apb @ xpy) / exci[None, :]
            X = ((xpy + xmy) * 0.5).T
            Y = ((xpy - xmy) * 0.5).T
        else:
            td = self._scf.TDDFT() if self.is_dft else self._scf.TDHF()
            td.nstates = nstates
            td.conv_tol = min(td.conv_tol, 1e-9)
            td.kernel()
            exci = td.e
            X = np.array([xy[0].ravel() for xy in td.xy])
            Y = np.array([xy[1].ravel() for xy in td.xy])
        norm = np.einsum('ni,ni->n', X, X) - np.einsum('ni,ni->n', Y, Y)
        X = X / np.sqrt(norm)[:, None]
        Y = Y / np.sqrt(norm)[:, None]
        self.exci = np.asarray(exci)
        self.X_vec = X.reshape(-1, o, v)
        self.Y_vec = Y.reshape(-1, o, v)
        return self.exci, self.X_vec, self.Y_vec

    def tpa_transition_tensor(self, n, resp_b, resp_c):
        '''Two-photon transition tensor T^N_BC = sqrt(2) z_N . r^BC(w1, w2) (Parker Eq. 35).'''
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

    def two_photon_absorption(self, nstates=5, states=None):
        '''Degenerate two-photon absorption (each photon w = Omega_N / 2).

        Returns exci, tensors (n, 3, 3) and rotationally averaged strengths (a.u.) for
        linearly ('linear') and circularly ('circular') polarized identical photons.
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
            resp = self.first_order(r, self.exci[n] * 0.5)
            tensors[k] = self.tpa_transition_tensor(n, resp, resp)
        delta_lin, delta_circ = _tpa_strength(tensors)
        self.tpa_tensor = tensors
        self.tpa_strength = {'linear': delta_lin, 'circular': delta_circ}
        log = logger.new_logger(self)
        log.note('Two-photon absorption (TDDFT quadratic response)')
        log.note('  state   Omega (eV)    delta_lin (a.u.)   delta_circ (a.u.)')
        for k, n in enumerate(states):
            log.note('  %5d  %10.4f  %16.6e  %16.6e', n + 1, self.exci[n] * 27.211386245988,
                     delta_lin[k], delta_circ[k])
        logger.timer(self, 'TDDFT two-photon absorption', *cput0)
        return self.exci[states], tensors, self.tpa_strength

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
    return (2 * dF + 2 * dG + 2 * dH) / 30.0, (-2 * dF + 3 * dG + 3 * dH) / 30.0


QRTDDFT = TDDFTQuadraticResponse
