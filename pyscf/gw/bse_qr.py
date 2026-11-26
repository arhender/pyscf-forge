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
* The BSE hyperkernel (second derivative of the self-energy w.r.t. G) is neglected, and so
  are all higher kernels; cubic response (second hyperpolarizability, three-photon
  absorption, see pyscf.tdscf.qr_base) therefore uses only G with fixed W.  No published
  BSE cubic-response reference exists to compare against.

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
import scipy.linalg

from pyscf import lib
from pyscf.lib import logger
from pyscf.gw.bse import BSE
from pyscf.gw.qp_shift import scissor_shift
from pyscf.tdscf.qr_base import (QuadraticResponseBase, beta_parallel, gamma_parallel,  # noqa: F401
                                 _tpa_strength, _tpa3_strength)

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
            hyperkernel g_xc of the Kohn-Sham reference functional to the second-order kernel
            (and g_xc plus the third-order kernel h_xc in cubic response), as in TDDFT.
            Default False.
        davidson_init_per_root : int
            Lower bound on the number of initial Davidson trial vectors, per requested
            excited state (bse.init_ntri >= davidson_init_per_root * nstates).  A larger
            initial space reduces the chance of missing states that couple weakly to the
            lowest orbital-energy-difference guesses.  Default 4.
        davidson_space_per_root : int
            Lower bound on the Davidson subspace size, per requested excited state
            (bse.max_vec >= init_ntri + davidson_space_per_root * nstates).  The full BSE
            Davidson adds up to 2*nstates vectors per iteration and stops with an error once
            the subspace is full, so this allows ~davidson_space_per_root/2 iterations.
            Capped by the memory left under max_memory.  Default 40.
        davidson_extra_roots : int
            Number of roots above nstates that the Davidson solver also converges and then
            discards.  States that couple weakly to the initial guess tend to enter the
            subspace late, just below the highest requested roots; tracking a buffer of
            extra roots keeps them from displacing converged ones (as in VeloxChem's
            num_extra_roots).  The init_ntri/max_vec scaling above uses
            nstates + davidson_extra_roots.  Default 2.

        Values already set on the underlying BSE object (self._bse.max_vec, .init_ntri,
        .max_expand) are only ever increased, never decreased.
    '''

    _label = 'BSE'
    davidson_init_per_root = 4
    davidson_space_per_root = 40
    davidson_extra_roots = 2

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
        # shared with the BSE/GW object (no copy unless it is not C-contiguous)
        self.Lpq = np.ascontiguousarray(bse.Lpq[0])
        if np.count_nonzero(self.mo_energy == 0.0) > 0:
            raise RuntimeError(
                'Some quasiparticle energies are exactly zero: GW was probably restricted to gw.orbs. '
                'Apply pyscf.gw.qp_shift.scissor_shift(gw, inplace=True) before building BSE, '
                'or pass the GW object (scissor=True) instead.')
        if self.Lpq.shape[1] != self.nmo or self.mo_coeff.shape[1] != self.nmo:
            raise NotImplementedError('Frozen orbitals are not supported in BSE quadratic response.')

        self.screening = screening
        self.xc_hyperkernel = False
        self._Lbar_oo = self._Lbar_ov = self._Lbar_vv = None
        self._xc_qr = None

    def _dump_method_flags(self, log):
        log.info('naux = %d', self.Lpq.shape[0])
        log.info('screened kernel (BSE) = %s', self.screening)
        log.info('BSE hyperkernel neglected (static screening)')
        if self.xc_hyperkernel:
            log.info('DIAGNOSTIC: reference XC hyperkernel added to second-order kernel (%s)',
                     getattr(self._bse.mf, 'xc', None))

    def build(self):
        '''Precompute the screened DF blocks used by the kernel contractions.

        Only the occupied-occupied, occupied-virtual and virtual-virtual blocks of
        Lbar = eps^-1 L (eps = 1 - Pi(0) in the auxiliary basis) are stored; each Lbar_P is
        symmetric in its orbital indices, so the virtual-occupied block is not needed.  The
        bare three-center integrals are always used through views of self.Lpq, so no other
        copy of the DF tensor is made.  Without screening the blocks are views of Lpq.
        '''
        o = self.nocc
        L = self.Lpq
        if self.screening:
            invd = _inverse_dielectric(self, L, self.mo_energy, o)
            self._Lbar_oo = _apply_aux(self, invd, L[:, :o, :o])
            self._Lbar_ov = _apply_aux(self, invd, L[:, :o, o:])
            self._Lbar_vv = _apply_aux(self, invd, L[:, o:, o:])
            del invd
        else:
            self._Lbar_oo, self._Lbar_ov, self._Lbar_vv = L[:, :o, :o], L[:, :o, o:], L[:, o:, o:]
        self._xc_qr = None
        if self.xc_hyperkernel:
            self._xc_qr = _reference_xc_hyperkernel(self)
        return QuadraticResponseBase.build(self)

    def _mem_words(self):
        '''Doubles available for temporaries: half of what is left under max_memory.'''
        avail = max(self.max_memory - lib.current_memory()[0], 100) * 0.5
        return int(avail * 1e6 / 8)

    def _aux_blksize(self, words_per_row):
        '''Number of auxiliary functions per block for temporaries of words_per_row doubles
        per auxiliary function.'''
        return max(1, min(self.Lpq.shape[0], self._mem_words() // max(int(words_per_row), 1)))

    def _lbar_full(self, p0, p1, out):
        '''Assemble Lbar[p0:p1] (all orbital pairs) from the stored blocks into out.'''
        o = self.nocc
        ov = self._Lbar_ov[p0:p1]
        out[:, :o, :o] = self._Lbar_oo[p0:p1]
        out[:, :o, o:] = ov
        out[:, o:, :o] = ov.transpose(0, 2, 1)
        out[:, o:, o:] = self._Lbar_vv[p0:p1]
        return out

    def _hyperkernel(self, Db, Dc):
        if not self.xc_hyperkernel:
            return None   # BSE hyperkernel neglected
        if self._xc_qr is None:
            self._xc_qr = _reference_xc_hyperkernel(self)
        return self._xc_qr.get_hyperkernel(Db, Dc)

    def _third_kernel(self, Db, Dc, Dd):
        if not self.xc_hyperkernel:
            return None
        if self._xc_qr is None:
            self._xc_qr = _reference_xc_hyperkernel(self)
        return self._xc_qr.get_third_kernel(Db, Dc, Dd)

    def get_kernel(self, dm):
        '''Kernel G[D] in the full MO space for one or several (possibly non-symmetric) D.

        G[D]_pq = 2 sum_rs (pq|rs) D_rs - sum_rs W(pr|sq) D_rs

        The exchange-type term is accumulated over blocks of auxiliary functions, one density
        at a time, so the temporaries are a few blocks of size (naux_blk, nmo, nmo).
        '''
        self._check_built()
        dm = np.asarray(dm, dtype=float)
        single = dm.ndim == 2
        if single:
            dm = dm[None]
        n = dm.shape[0]
        L = self.Lpq
        naux, nmo = L.shape[:2]
        L2 = L.reshape(naux, nmo * nmo)
        rho = L2 @ dm.reshape(n, nmo * nmo).T                  # (naux, n)
        out = (rho.T @ L2).reshape(n, nmo, nmo)
        out *= 2.0
        # Lbar block, T and einsum's transposed copy of T: about 3 nmo^2 per aux function
        blk = self._aux_blksize(3 * nmo * nmo)
        wbuf = np.empty((blk, nmo, nmo))
        for p0, p1 in lib.prange(0, naux, blk):
            W = self._lbar_full(p0, p1, wbuf[:p1 - p0])
            for k in range(n):
                T = einsum('Ppr,rs->Pps', L[p0:p1], dm[k])
                out[k] -= einsum('Pps,Psq->pq', T, W)
                del T
        return out[0] if single else out

    def _bse_terms(self, z):
        '''Occupied-virtual BSE contractions for trial vectors z of shape (n, nocc, nvir):

            vz_ia = sum_jb (ia|jb) z_jb
            wa_ia = -sum_jb W(ij|ab) z_jb
            wb_ia = -sum_jb W(ib|ja) z_jb

        so that (A + B) z = d z + 4 vz + wa + wb and (A - B) z = d z + wa - wb (singlet; the
        same terms as pyscf.gw.bse._bse_contraction).  Accumulated over auxiliary blocks on
        views of Lpq.
        '''
        n, o, v = z.shape
        L = self.Lpq
        vz = np.zeros((n, o, v))
        wa = np.zeros((n, o, v))
        wb = np.zeros((n, o, v))
        zt = z.transpose(0, 2, 1)
        blk = self._aux_blksize(4 * o * v + 3 * o * o)
        for p0, p1 in lib.prange(0, L.shape[0], blk):
            b = p1 - p0
            Lov = np.ascontiguousarray(L[p0:p1, :o, o:]).reshape(b, o * v)
            Lvv = L[p0:p1, o:, o:]                                         # view
            Boo = self._Lbar_oo[p0:p1].transpose(1, 0, 2).reshape(o, b * o)  # [i, (P, j)]
            Bov = self._Lbar_ov[p0:p1].reshape(b * o, v)                   # [(P, j), a]
            rho = Lov @ z.reshape(n, o * v).T
            vz += (rho.T @ Lov).reshape(n, o, v)
            Lov = Lov.reshape(b, o, v)
            for k in range(n):
                T = np.matmul(z[k], Lvv)                    # T[P, j, a] = sum_b z_jb L_P,ba
                wa[k] -= Boo @ T.reshape(b * o, v)
                U = np.matmul(Lov, zt[k])                   # U[P, i, j] = sum_b L_P,ib z_jb
                wb[k] -= U.transpose(1, 0, 2).reshape(o, b * o) @ Bov
        return vz, wa, wb

    def apb_amb_prod(self, zp, zm):
        '''Return (A+B) zp and (A-B) zm for vectors in the (occ, vir) flattened space.'''
        self._check_built()
        o = self.nocc
        v = self.nmo - o
        d = self._orb_diff()
        vz, wa, wb = self._bse_terms(np.stack([zp.reshape(o, v), zm.reshape(o, v)]))
        apb = d * zp.ravel() + (4.0 * vz[0] + wa[0] + wb[0]).ravel()
        amb = d * zm.ravel() + (wa[1] - wb[1]).ravel()
        return apb, amb

    def _lr_matvec(self, omega, X, Y):
        '''[[A - w, B], [B, A + w]] (X, Y) with A z = d z + 2 vz + wa, B z = 2 vz + wb.'''
        self._check_built()
        vz, wa, wb = self._bse_terms(np.stack([X, Y]))
        d = self._orb_diff().reshape(X.shape)
        v2 = 2.0 * (vz[0] + vz[1])
        return ((d - omega) * X + v2 + wa[0] + wb[1],
                (d + omega) * Y + v2 + wa[1] + wb[0])

    def get_dense_apb_amb(self):
        '''Dense A+B and A-B matrices (dimension nocc*nvir), built directly from the DF
        blocks (two dim x dim arrays plus small temporaries).'''
        if self._apb is None:
            self._check_built()
            o = self.nocc
            v = self.nmo - o
            dim = o * v
            L = self.Lpq
            apb = np.zeros((dim, dim))
            amb = np.zeros((dim, dim))
            apb4 = apb.reshape(o, v, o, v)
            amb4 = amb.reshape(o, v, o, v)
            blk = self._aux_blksize(2 * dim + v * v)
            for p0, p1 in lib.prange(0, L.shape[0], blk):
                b = p1 - p0
                Lov = np.ascontiguousarray(L[p0:p1, :o, o:]).reshape(b, dim)
                lib.dot(Lov.T, Lov, 4.0, apb, 1.0)                      # 4 (ia|jb)
                Lvv = np.ascontiguousarray(L[p0:p1, o:, o:]).reshape(b, v * v)
                Bov = np.ascontiguousarray(self._Lbar_ov[p0:p1]).reshape(b, dim)
                Lov = Lov.reshape(b, o, v)
                for i in range(o):
                    # W(ij|ab) = sum_P Lbar_P,ij L_P,ab  -> [j, a, b]
                    wa = (self._Lbar_oo[p0:p1, i, :].T @ Lvv).reshape(o, v, v)
                    # W(ib|ja) = sum_P Lbar_P,ja L_P,ib  -> [(j, a), b]
                    wb = (Bov.T @ Lov[:, i, :]).reshape(o, v, v)
                    wa = wa.transpose(1, 0, 2)                           # [a, j, b]
                    wb = wb.transpose(1, 0, 2)
                    apb4[i] -= wa
                    apb4[i] -= wb
                    amb4[i] -= wa
                    amb4[i] += wb
            d = self._orb_diff()
            idx = np.arange(dim)
            apb[idx, idx] += d
            amb[idx, idx] += d
            self._set_dense(apb, amb)
        return self._apb, self._amb

    def _size_davidson(self, bse, nstates):
        '''Scale the BSE Davidson initial space, expansion and subspace size with nstates.

        BSE.__init__ sizes these from its default nroot (10), so setting bse.nroot later
        leaves them unchanged; a too-small subspace makes bse_davidson raise
        "Exceeded max_vec" when late-appearing roots (weakly coupled to the initial guess)
        force extra iterations.
        '''
        log = logger.new_logger(self)
        full_dim = int(sum(bse.nocc[s] * (bse.nmo - bse.nocc[s]) for s in range(bse.nspin)))

        init_ntri = min(full_dim, max(bse.init_ntri, self.davidson_init_per_root * nstates))
        max_expand = max(bse.max_expand, min(100, nstates))
        want = max(bse.max_vec, init_ntri + self.davidson_space_per_root * nstates)
        # tri_vec, apb_prod and amb_prod each hold max_vec rows of length full_dim
        avail = (self.max_memory - lib.current_memory()[0]) * 1e6 / (3 * 8 * max(full_dim, 1))
        floor = max(bse.max_vec, 12 * nstates, init_ntri + 2 * nstates)
        max_vec = want
        if want > max(avail, floor):
            max_vec = int(max(avail, floor))
            log.warn('BSE Davidson subspace limited to %d vectors by max_memory (wanted %d).',
                     max_vec, want)
        # the subspace can never exceed the full space (+ one expansion step of slack)
        max_vec = int(min(max_vec, full_dim + 2 * max_expand))

        bse.init_ntri = int(init_ntri)
        bse.max_expand = int(max_expand)
        bse.max_vec = max_vec
        log.info('BSE Davidson: nroot = %d  init_ntri = %d  max_expand = %d  max_vec = %d',
                 nstates, bse.init_ntri, bse.max_expand, bse.max_vec)

    def _excited_states_large(self, nstates):
        if not self.screening:
            raise NotImplementedError('Use solver="direct" for unscreened excited states.')
        bse = self._bse
        dim = self.nocc * (self.nmo - self.nocc)
        nroot = min(dim, nstates + max(0, int(self.davidson_extra_roots)))
        bse.nroot = nroot
        self._size_davidson(bse, nroot)
        exci, X_vec, Y_vec = bse.kernel('s')
        if len(exci) < nstates:
            raise RuntimeError('BSE Davidson returned %d roots, %d requested.' % (len(exci), nstates))
        X = X_vec[0].reshape(-1, dim)[:nstates]
        Y = Y_vec[0].reshape(-1, dim)[:nstates]
        return np.asarray(exci)[:nstates], X, Y



def _inverse_dielectric(qr, L, mo_energy, nocc):
    '''eps^-1 = (1 - Pi(0))^-1 in the auxiliary basis (restricted, spin-summed static
    response), accumulated over blocks of occupied orbitals.'''
    naux, nmo = L.shape[:2]
    nvir = nmo - nocc
    eps = np.zeros((naux, naux))
    # one occupied orbital needs 2 * naux * nvir doubles (Lia chunk + Pia)
    iblk = max(1, min(nocc, qr._mem_words() // max(2 * naux * nvir, 1)))
    for i0, i1 in lib.prange(0, nocc, iblk):
        Lia = np.ascontiguousarray(L[:, i0:i1, nocc:]).reshape(naux, -1)
        eia = 4.0 / (mo_energy[i0:i1, None] - mo_energy[None, nocc:])
        Pia = Lia * eia.reshape(1, -1)
        lib.dot(Pia, Lia.T, -1.0, eps, 1.0)
    eps[np.diag_indices(naux)] += 1.0
    return scipy.linalg.inv(eps, overwrite_a=True, check_finite=False)


def _apply_aux(qr, invd, Lblk):
    '''invd @ Lblk for a (naux, n1, n2) view of the DF tensor, in chunks of the first
    orbital index (contiguous result).'''
    naux, n1, n2 = Lblk.shape
    out = np.empty((naux, n1, n2))
    # one orbital row needs 2 * naux * n2 doubles (copy of the input chunk + product)
    rblk = max(1, min(n1, qr._mem_words() // max(2 * naux * n2, 1)))
    for r0, r1 in lib.prange(0, n1, rblk):
        chunk = np.ascontiguousarray(Lblk[:, r0:r1]).reshape(naux, -1)
        out[:, r0:r1] = (invd @ chunk).reshape(naux, r1 - r0, n2)
    return out


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
