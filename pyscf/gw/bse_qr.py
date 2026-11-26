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
from pyscf.gw.bse_lowmem import transform_df_pairs, _SpinLpq
from pyscf.gw.qp_shift import scissor_shift
from pyscf.tdscf.qr_base import (QuadraticResponseBase, beta_parallel, gamma_parallel,  # noqa: F401
                                 _tpa_strength, _tpa3_strength)

einsum = lib.einsum


class _DFBlockStore:
    '''Occupied-occupied, occupied-virtual and virtual-virtual blocks of the MO DF integrals
    L[P, p, q] (bare) and of the screened Lbar = eps^-1 L, aux-major.

    Each block is an in-memory array or, if it does not fit the memory budget, an h5py
    dataset in a temporary file under lib.param.TMPDIR; all access goes through slices of
    the auxiliary index (get) or of the first orbital index (rows), so the kernels read the
    blocks in pieces either way.  Without screening the screened blocks are the bare ones.
    '''

    NAMES = ('oo', 'ov', 'vv')

    def __init__(self, naux, nocc, nvir):
        self.naux = naux
        self.o = nocc
        self.v = nvir
        self.bare = {}
        self.scr = {}
        self.screened_for = None    # value of `screening` the screened blocks were built for
        self._h5 = None

    def shape(self, name):
        n1 = self.o if name[0] == 'o' else self.v
        n2 = self.o if name[1] == 'o' else self.v
        return (self.naux, n1, n2)

    def nbytes(self, name):
        return int(np.prod(self.shape(name))) * 8

    def alloc(self, name, incore):
        shape = self.shape(name)
        if incore:
            return np.empty(shape)
        if self._h5 is None:
            self._h5 = lib.H5TmpFile(dir=lib.param.TMPDIR)
        n2 = max(shape[2], 1)
        chunks = (min(shape[0], 8), max(1, min(shape[1], (1 << 16) // n2)), shape[2])
        key = '%s_%d' % (name, len(self._h5))
        return self._h5.create_dataset(key, shape, 'f8', chunks=chunks)

    def _blocks(self, screened):
        return self.scr if screened else self.bare

    def get(self, name, p0, p1, screened=False):
        '''Block [p0:p1] of the auxiliary index as an ndarray (a view if in memory).'''
        return np.asarray(self._blocks(screened)[name][p0:p1])

    def rows(self, name, r0, r1, screened=False):
        '''All auxiliary functions, first orbital index r0:r1.'''
        return np.asarray(self._blocks(screened)[name][:, r0:r1])

    def full(self, p0, p1, screened, out):
        '''Assemble L[p0:p1] (all orbital pairs) into out (L_P is symmetric).'''
        o = self.o
        ov = self.get('ov', p0, p1, screened)
        out[:, :o, :o] = self.get('oo', p0, p1, screened)
        out[:, :o, o:] = ov
        out[:, o:, :o] = ov.transpose(0, 2, 1)
        out[:, o:, o:] = self.get('vv', p0, p1, screened)
        return out

    def ondisk(self):
        return ['%s%s' % ('W' if s else 'L', k) for s, d in ((0, self.bare), (1, self.scr))
                for k, b in d.items() if not isinstance(b, np.ndarray)]


def _place_blocks(store, budget_mb, screening):
    '''Decide which blocks stay in memory: greedily, in the order bare oo, ov, screened oo,
    ov, bare vv, screened vv, while the budget allows.  Returns {(name, screened): incore}.'''
    order = [('oo', 0), ('ov', 0), ('oo', 1), ('ov', 1), ('vv', 0), ('vv', 1)]
    left = budget_mb * 1e6
    out = {}
    for name, s in order:
        if s and not screening:
            continue
        size = store.nbytes(name)
        out[(name, s)] = size <= left
        if size <= left:
            left -= size
    return out


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
        incore : None or bool
            Where the DF integral blocks (bare and screened oo/ov/vv) are kept.  None
            (default): in memory as long as they fit max_memory minus a working reserve,
            largest (virtual-virtual) blocks on disk otherwise; True: all in memory; False:
            all on disk (lib.param.TMPDIR, e.g. $SLURM_TMPDIR).  The full MO DF tensor is never
            formed when the integrals come from a density-fitting object.
        davidson_init_per_root : int
            Initial trial vectors (lowest orbital-energy differences) per tracked root of the
            iterative excited-state solver.  A larger initial space reduces the chance of
            missing states that couple weakly to the lowest guesses.  Default 4.
        davidson_space_per_root : int
            Subspace size of the excited-state solver, per tracked root, on top of the
            initial vectors; when it is full the solver restarts from the current Ritz
            vectors (pyscf.tdscf._lr_eig.real_eig).  Capped by max_memory.  Default 40.
        davidson_extra_roots : int
            Number of roots above nstates that the excited-state solver also converges and
            then discards.  States that couple weakly to the initial guess tend to enter the
            subspace late, just below the highest requested roots; tracking a buffer of extra
            roots keeps them from displacing requested ones (as in VeloxChem's
            num_extra_roots).  Default 2.
        davidson_conv_tol : float
            Residual-norm threshold of the excited-state solver.  Default 1e-5.
        davidson_max_cycle : int
            Maximum iterations of the excited-state solver.  Default 100.
    '''

    _label = 'BSE'
    davidson_init_per_root = 4
    davidson_space_per_root = 40
    davidson_extra_roots = 2
    davidson_conv_tol = 1e-5
    davidson_max_cycle = 100
    # memory kept free for temporaries when placing the integral blocks (fraction, MB floor)
    _reserve_frac = 0.25
    _reserve_min_mb = 2000

    def __init__(self, obj, screening=True, scissor=True, incore=None):
        '''Initialize from a BSE object, a GW object (GWAC, GWCD, GWFSCD, ...), or a
        restricted mean-field object.

        Passing a mean-field object uses its orbital energies instead of GW quasiparticle
        energies (useful for testing: with screening=False and an RHF reference this is TDHF).

        If a GW object only corrected the orbitals in gw.orbs, the remaining orbitals are
        filled in by a scissor shift (see pyscf.gw.qp_shift.scissor_shift) unless
        scissor=False.  The GW object itself is not modified.  A BSE object must already
        carry complete quasiparticle energies; apply scissor_shift to the GW object first.

        Integrals: a full MO DF tensor already held by the BSE/GW object (Lpq) is used
        through views; otherwise (e.g. GWFSCD with lowmem_ints, BSELowMem, mean-field input)
        only the oo/ov/vv blocks are built, from the blocks of a BSELowMem or directly from
        the density-fitting object, in memory or on disk according to `incore`.
        '''
        bse = None
        L = None
        with_df = None
        if isinstance(obj, BSE):
            bse = obj
            if bse.nspin != 1:
                raise NotImplementedError('BSE quadratic response is implemented for restricted references only.')
            mf, mol = bse.mf, bse.mol
            mo_energy, mo_coeff, nocc = bse.mo_energy[0], bse.mo_coeff[0], int(bse.nocc[0])
            verbose, stdout = bse.verbose, bse.stdout
            L = bse.Lpq[0]
        elif hasattr(obj, '_scf') and hasattr(obj, 'mo_energy'):
            gw = obj
            if getattr(gw, 'frozen', None) is not None:
                raise NotImplementedError('Frozen orbitals are not supported in BSE quadratic response.')
            if np.asarray(gw.mo_energy).ndim != 1:
                raise NotImplementedError('BSE quadratic response is implemented for restricted references only.')
            mf, mol = gw._scf, gw.mol
            mo_energy = scissor_shift(gw) if scissor else np.asarray(gw.mo_energy)
            mo_coeff, nocc = np.asarray(gw.mo_coeff), int(gw.nocc)
            verbose, stdout = gw.verbose, gw.stdout
            L = getattr(gw, 'Lpq', None)
            if L is not None and np.ndim(L) == 4:
                L = L[0]
            if not (isinstance(L, np.ndarray) and L.ndim == 3):
                L = None
                if getattr(gw, 'with_df', None) is None:
                    gw.initialize_df(auxbasis=getattr(gw, 'auxbasis', None))
                with_df = gw.with_df
        elif hasattr(obj, 'mo_energy') and hasattr(obj, 'mol'):
            mf = obj
            if np.asarray(mf.mo_energy).ndim != 1:
                raise NotImplementedError('BSE quadratic response is implemented for restricted references only.')
            from pyscf.gw.gw_ac import GWAC
            gw = GWAC(mf)
            gw.initialize_df()
            mol, mo_energy, mo_coeff = mf.mol, np.asarray(mf.mo_energy), np.asarray(mf.mo_coeff)
            nocc = int(np.count_nonzero(mf.mo_occ > 0))
            verbose, stdout = mf.verbose, mf.stdout
            with_df = gw.with_df
        else:
            raise TypeError('Expected a BSE, GW or restricted mean-field object, got %s' % type(obj))

        QuadraticResponseBase.__init__(
            self, mol, mo_energy, mo_coeff, nocc, verbose=verbose, stdout=stdout,
            max_memory=getattr(mf, 'max_memory', mol.max_memory))
        self._bse = bse
        self._mf = mf
        if np.count_nonzero(self.mo_energy == 0.0) > 0:
            raise RuntimeError(
                'Some quasiparticle energies are exactly zero: GW was probably restricted to gw.orbs. '
                'Apply pyscf.gw.qp_shift.scissor_shift(gw, inplace=True) before building BSE, '
                'or pass the GW object (scissor=True) instead.')
        if self.mo_coeff.shape[1] != self.nmo or (L is not None and L.shape[1] != self.nmo):
            raise NotImplementedError('Frozen orbitals are not supported in BSE quadratic response.')

        self.screening = screening
        self.incore = incore
        self.xc_hyperkernel = False
        self._xc_qr = None
        self._store = self._init_store(L, with_df)
        self.naux = self._store.naux

    # ------------------------------------------------------------ integral blocks

    def _block_budget_mb(self):
        '''Memory (MB) available for in-memory integral blocks.'''
        left = self.max_memory - lib.current_memory()[0]
        return left - max(self._reserve_frac * self.max_memory, self._reserve_min_mb)

    def _placement(self, store):
        if self.incore is None:
            return _place_blocks(store, self._block_budget_mb(), self.screening)
        keys = [(n, s) for n in store.NAMES for s in (0, 1) if self.screening or not s]
        return {k: bool(self.incore) for k in keys}

    def _init_store(self, L, with_df):
        '''Bare oo/ov/vv blocks: views of a full in-memory tensor, adopted BSELowMem blocks,
        or transformed from the DF object into memory/disk targets.'''
        o, n = self.nocc, self.nmo
        if L is not None and isinstance(L, _SpinLpq):
            store = _DFBlockStore(L.naux, o, n - o)
            store.bare = dict(L.b)
            return store
        if L is not None:
            L = np.asarray(L)
            store = _DFBlockStore(L.shape[0], o, n - o)
            store.bare = {'oo': L[:, :o, :o], 'ov': L[:, :o, o:], 'vv': L[:, o:, o:]}
            return store
        naux = with_df.get_naoaux()
        store = _DFBlockStore(naux, o, n - o)
        place = self._placement(store)
        targets = {k: store.alloc(k, place[(k, 0)]) for k in store.NAMES}
        pairs = {'oo': (slice(0, o), slice(0, o)), 'ov': (slice(0, o), slice(o, n)),
                 'vv': (slice(o, n), slice(o, n))}
        cput0 = (logger.process_clock(), logger.perf_counter())
        with lib.temporary_env(with_df, verbose=0):
            store.bare = transform_df_pairs(with_df, self.mo_coeff, pairs, self.max_memory, out=targets)
        logger.timer(self, 'BSE DF integral blocks', *cput0)
        return store

    @property
    def Lpq(self):
        '''Full MO DF tensor (naux, nmo, nmo), assembled on demand (small systems/tests only).'''
        out = np.empty((self.naux, self.nmo, self.nmo))
        return self._store.full(0, self.naux, False, out)

    def _dump_method_flags(self, log):
        log.info('naux = %d', self.naux)
        log.info('screened kernel (BSE) = %s', self.screening)
        log.info('BSE hyperkernel neglected (static screening)')
        disk = self._store.ondisk()
        log.info('DF integral blocks on disk: %s', ', '.join(disk) if disk else 'none')
        if self.xc_hyperkernel:
            log.info('DIAGNOSTIC: reference XC hyperkernel added to second-order kernel (%s)',
                     getattr(self._mf, 'xc', None))

    def build(self):
        '''Precompute the screened DF blocks used by the kernel contractions.

        Only the occupied-occupied, occupied-virtual and virtual-virtual blocks of
        Lbar = eps^-1 L (eps = 1 - Pi(0) in the auxiliary basis) are stored; each Lbar_P is
        symmetric in its orbital indices, so the virtual-occupied block is not needed.  They
        go to memory or disk like the bare blocks (see `incore`).  Without screening the
        screened blocks are the bare ones.
        '''
        if self._store.screened_for != self.screening:
            self._build_screened()
        self._xc_qr = None
        if self.xc_hyperkernel:
            self._xc_qr = _reference_xc_hyperkernel(self)
        return QuadraticResponseBase.build(self)

    def _build_screened(self):
        store = self._store
        if not self.screening:
            store.scr = dict(store.bare)
        else:
            cput0 = (logger.process_clock(), logger.perf_counter())
            invd = _inverse_dielectric(self, store, self.mo_energy, self.nocc)
            place = self._placement(store)
            scr = {}
            for name in store.NAMES:
                scr[name] = store.alloc(name, place[(name, 1)])
                _apply_aux(self, invd, store, name, scr[name])
            del invd
            store.scr = scr
            logger.timer(self, 'BSE screened DF blocks', *cput0)
        store.screened_for = self.screening

    def _mem_words(self):
        '''Doubles available for temporaries: half of what is left under max_memory.'''
        avail = max(self.max_memory - lib.current_memory()[0], 100) * 0.5
        return int(avail * 1e6 / 8)

    def _aux_blksize(self, words_per_row):
        '''Number of auxiliary functions per block for temporaries of words_per_row doubles
        per auxiliary function.'''
        return max(1, min(self.naux, self._mem_words() // max(int(words_per_row), 1)))

    def _lbar_full(self, p0, p1, out):
        '''Assemble Lbar[p0:p1] (all orbital pairs) from the stored blocks into out.'''
        return self._store.full(p0, p1, True, out)

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

        Both terms are accumulated over blocks of auxiliary functions (each block of the
        bare and screened integrals is read once for all densities), so the temporaries are a
        few blocks of size (naux_blk, nmo, nmo).
        '''
        self._check_built()
        dm = np.asarray(dm, dtype=float)
        single = dm.ndim == 2
        if single:
            dm = dm[None]
        n = dm.shape[0]
        nmo = self.nmo
        store = self._store
        out = np.zeros((n, nmo, nmo))
        dm2 = dm.reshape(n, nmo * nmo)
        # L and Lbar blocks, T and einsum's transposed copy of T: about 4 nmo^2 per aux function
        blk = self._aux_blksize(4 * nmo * nmo)
        lbuf = np.empty((blk, nmo, nmo))
        wbuf = np.empty((blk, nmo, nmo))
        for p0, p1 in lib.prange(0, self.naux, blk):
            b = p1 - p0
            L = store.full(p0, p1, False, lbuf[:b])
            W = store.full(p0, p1, True, wbuf[:b])
            L2 = L.reshape(b, nmo * nmo)
            rho = L2 @ dm2.T                                   # (b, n)
            out += 2.0 * (rho.T @ L2).reshape(n, nmo, nmo)
            for k in range(n):
                T = einsum('Ppr,rs->Pps', L, dm[k])
                out[k] -= einsum('Pps,Psq->pq', T, W)
                del T
        return out[0] if single else out

    def _bse_terms(self, z):
        '''Occupied-virtual BSE contractions for trial vectors z of shape (n, nocc, nvir):

            vz_ia = sum_jb (ia|jb) z_jb
            wa_ia = -sum_jb W(ij|ab) z_jb
            wb_ia = -sum_jb W(ib|ja) z_jb

        so that (A + B) z = d z + 4 vz + wa + wb and (A - B) z = d z + wa - wb (singlet; the
        same terms as pyscf.gw.bse._bse_contraction).  Accumulated over auxiliary blocks; each
        block of the integrals is read once for all n vectors.
        '''
        n, o, v = z.shape
        store = self._store
        vz = np.zeros((n, o, v))
        wa = np.zeros((n, o, v))
        wb = np.zeros((n, o, v))
        zt = z.transpose(0, 2, 1)
        # per aux function: L_ov, L_vv (a copy if on disk), Lbar_oo/ov (copies if on disk),
        # T and U temporaries
        blk = self._aux_blksize(4 * o * v + 3 * o * o + v * v)
        for p0, p1 in lib.prange(0, self.naux, blk):
            b = p1 - p0
            Lov = np.ascontiguousarray(store.get('ov', p0, p1)).reshape(b, o * v)
            Lvv = store.get('vv', p0, p1)
            Boo = store.get('oo', p0, p1, True).transpose(1, 0, 2).reshape(o, b * o)  # [i, (P, j)]
            Bov = np.ascontiguousarray(store.get('ov', p0, p1, True)).reshape(b * o, v)  # [(P, j), a]
            rho = Lov @ z.reshape(n, o * v).T
            vz += (rho.T @ Lov).reshape(n, o, v)
            Lov = Lov.reshape(b, o, v)
            for k in range(n):
                T = np.matmul(z[k], Lvv)                    # T[P, j, a] = sum_b z_jb L_P,ba
                wa[k] -= Boo @ T.reshape(b * o, v)
                U = np.matmul(Lov, zt[k])                   # U[P, i, j] = sum_b L_P,ib z_jb
                wb[k] -= U.transpose(1, 0, 2).reshape(o, b * o) @ Bov
            del Lov, Lvv, Boo, Bov
        return vz, wa, wb

    def _apb_amb_batch(self, zp, zm):
        '''(A+B) zp and (A-B) zm for stacks of vectors (np, dim) and (nm, dim), in one pass
        over the integrals.'''
        self._check_built()
        o = self.nocc
        v = self.nmo - o
        zp = np.asarray(zp).reshape(-1, o * v)
        zm = np.asarray(zm).reshape(-1, o * v)
        npl = zp.shape[0]
        d = self._orb_diff()
        vz, wa, wb = self._bse_terms(np.vstack([zp, zm]).reshape(-1, o, v))
        dim = o * v
        apb = d * zp + (4.0 * vz[:npl] + wa[:npl] + wb[:npl]).reshape(npl, dim)
        amb = d * zm + (wa[npl:] - wb[npl:]).reshape(-1, dim)
        return apb, amb

    def apb_amb_prod(self, zp, zm):
        '''Return (A+B) zp and (A-B) zm for vectors in the (occ, vir) flattened space.'''
        apb, amb = self._apb_amb_batch(np.ravel(zp)[None], np.ravel(zm)[None])
        return apb[0], amb[0]

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
            store = self._store
            apb = np.zeros((dim, dim))
            amb = np.zeros((dim, dim))
            apb4 = apb.reshape(o, v, o, v)
            amb4 = amb.reshape(o, v, o, v)
            blk = self._aux_blksize(2 * dim + v * v + o * o)
            for p0, p1 in lib.prange(0, self.naux, blk):
                b = p1 - p0
                Lov = np.ascontiguousarray(store.get('ov', p0, p1)).reshape(b, dim)
                lib.dot(Lov.T, Lov, 4.0, apb, 1.0)                      # 4 (ia|jb)
                Lvv = np.ascontiguousarray(store.get('vv', p0, p1)).reshape(b, v * v)
                Bov = np.ascontiguousarray(store.get('ov', p0, p1, True)).reshape(b, dim)
                Boo = store.get('oo', p0, p1, True)
                Lov = Lov.reshape(b, o, v)
                for i in range(o):
                    # W(ij|ab) = sum_P Lbar_P,ij L_P,ab  -> [j, a, b]
                    wa = (Boo[:, i, :].T @ Lvv).reshape(o, v, v)
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

    def _excited_states_large(self, nstates):
        '''Lowest singlet BSE excitations with the restarted Davidson solver of
        pyscf.tdscf._lr_eig.real_eig on the integral-block matrix-vector products.  Tracks
        nstates + davidson_extra_roots roots and returns the lowest nstates.'''
        from pyscf.tdscf import _lr_eig
        if not self.screening:
            raise NotImplementedError('Use solver="direct" for unscreened excited states.')
        self._check_built()
        log = logger.new_logger(self)
        dim = self.nocc * (self.nmo - self.nocc)
        d = self._orb_diff()
        nroot = min(dim, nstates + max(0, int(self.davidson_extra_roots)))
        ninit = min(dim, max(nroot, self.davidson_init_per_root * nroot))
        max_space = min(dim, ninit + self.davidson_space_per_root * nroot)
        # new trial vectors per iteration only for the tracked roots (real_eig's default of
        # up to 20 would multiply the cost of each pass over the integrals)
        space_inc = nroot
        # real_eig keeps 4 (dim, max_space) holders: max_space = mem / (8 * 4 dim) / 2 - space_inc
        mem_mb = (max_space + space_inc) * 2 * 4 * dim * 8 / 1e6
        avail = 0.5 * (self.max_memory - lib.current_memory()[0])
        if mem_mb > avail:
            log.warn('BSE excited-state subspace limited by max_memory (%.0f MB wanted, %.0f MB free)',
                     mem_mb, avail)
            mem_mb = max(avail, 4 * nroot * 2 * 4 * dim * 8 / 1e6)
        log.info('BSE Davidson (real_eig): nroot = %d (%d requested)  initial = %d  max_space = %d',
                 nroot, nstates, ninit, max_space)

        x0 = np.zeros((ninit, 2 * dim))
        x0[np.arange(ninit), np.argsort(d, kind='stable')[:ninit]] = 1.0
        diag = np.concatenate([d, -d])

        def aop(xs):
            xs = np.asarray(xs)
            X, Y = xs[:, :dim], xs[:, dim:]
            P, M = self._apb_amb_batch(X + Y, X - Y)
            return np.hstack([0.5 * (P + M), -0.5 * (P - M)])

        def precond(dx, e):
            den = diag[None, :] - np.asarray(e)[:, None]
            den[np.abs(den) < 1e-8] = 1e-8
            return dx / den

        cput0 = (logger.process_clock(), logger.perf_counter())
        with lib.temporary_env(_lr_eig, MAX_SPACE_INC=None):
            conv, e, xy = _lr_eig.real_eig(aop, x0, precond, tol_residual=self.davidson_conv_tol,
                                           nroots=nroot, max_cycle=self.davidson_max_cycle,
                                           max_memory=mem_mb, verbose=log)
        logger.timer(self, 'BSE Davidson', *cput0)
        conv = np.asarray(conv)
        if len(e) < nstates or not np.all(conv[:nstates]):
            raise RuntimeError('BSE excited states not converged (converged: %s, energies %s)'
                               % (conv, e))
        if not np.all(conv):
            log.warn('BSE Davidson: extra (buffer) roots not converged: %s', conv)
        xy = np.asarray(xy)
        return np.asarray(e)[:nstates], xy[:nstates, :dim], xy[:nstates, dim:]


def _inverse_dielectric(qr, store, mo_energy, nocc):
    '''eps^-1 = (1 - Pi(0))^-1 in the auxiliary basis (restricted, spin-summed static
    response), accumulated over blocks of occupied orbitals of the bare ov block.'''
    naux, nvir = store.naux, store.v
    eps = np.zeros((naux, naux))
    # one occupied orbital needs 2 * naux * nvir doubles (Lia chunk + Pia)
    iblk = max(1, min(nocc, qr._mem_words() // max(2 * naux * nvir, 1)))
    for i0, i1 in lib.prange(0, nocc, iblk):
        Lia = np.ascontiguousarray(store.rows('ov', i0, i1)).reshape(naux, -1)
        eia = 4.0 / (mo_energy[i0:i1, None] - mo_energy[None, nocc:])
        Pia = Lia * eia.reshape(1, -1)
        lib.dot(Pia, Lia.T, -1.0, eps, 1.0)
    eps[np.diag_indices(naux)] += 1.0
    return scipy.linalg.inv(eps, overwrite_a=True, check_finite=False)


def _apply_aux(qr, invd, store, name, out):
    '''out = invd @ L_name for the bare block `name` of the store, in chunks of the first
    orbital index (out is an array or an h5py dataset).'''
    naux, n1, n2 = store.shape(name)
    # one orbital row needs 2 * naux * n2 doubles (copy of the input chunk + product)
    rblk = max(1, min(n1, qr._mem_words() // max(2 * naux * n2, 1)))
    for r0, r1 in lib.prange(0, n1, rblk):
        chunk = np.ascontiguousarray(store.rows(name, r0, r1)).reshape(naux, -1)
        out[:, r0:r1] = (invd @ chunk).reshape(naux, r1 - r0, n2)
    return out


def _reference_xc_hyperkernel(qr):
    '''TDDFT helper evaluating the reference functional's XC hyperkernel on the BSE orbitals.'''
    from pyscf.tdscf.quadratic_response import TDDFTQuadraticResponse
    mf = qr._mf
    if not np.allclose(np.asarray(mf.mo_coeff), qr.mo_coeff):
        raise RuntimeError('xc_hyperkernel requires the BSE orbitals to be the reference KS orbitals.')
    helper = TDDFTQuadraticResponse(mf)
    helper.verbose = 0
    helper.build()
    if helper._xc is None:
        logger.warn(qr, 'xc_hyperkernel: reference has no semilocal XC part; the term is zero.')
    return helper


QRBSE = BSEQuadraticResponse
