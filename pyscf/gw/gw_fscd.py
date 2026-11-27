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
Spin-restricted G0W0 with frequency-sampled contour deformation (fsCD-GW).

Canonical contour deformation (pyscf.gw.gw_cd) splits the correlation self-energy into an
imaginary-axis integral and a sum of residues,

    Sigma^c_p(w) = I^c_p(w) + R^c_p(w),
    R^c_p(w) = sum_m f_m(w) W^c_{pm,pm}(|w - e_m| + i eta),

where f_m = +1 for e_F < e_m < w and -1 for w < e_m < e_F.  Every residue needs the
dielectric matrix at its own real frequency, so a core state costs nocc * O(N^4) = O(N^5)
per evaluation of the self-energy.

fsCD-GW (Kehry, Klopper, Holzer, J. Chem. Phys. 159, 044116 (2023); building on Duchemin and
Blase, J. Chem. Theory Comput. 16, 1742 (2020)) evaluates W^c only on a small set of sample
frequencies and represents each needed diagonal element W^c_{pm,pm}(z) by a rational function
fitted with the AAA algorithm.  The sample set is

  * the Gauss-Legendre imaginary frequencies of the I^c integral (W already available there),
  * a sparse grid on the real axis, built from all residue frequencies |e_p - e_m| by
    discarding points closer than T1 (absolute) or T2 (relative) to an already kept point;
    the residues of different shells form tight clusters, so only a handful of real
    frequencies remain.

Residues closer than 5e-6 a.u. to a sample frequency use the sampled W directly.  The cost of
R^c is then essentially independent of the number of residues, i.e. O(N^4) overall.

Implementation notes:
  * The real-axis grid is first built from the residues at the mean-field energies and is
    then extended on the fly: whenever the self-energy is evaluated at a frequency w (e.g. a
    Newton iterate of the QP equation) whose residues are not represented under the T1/T2
    rule, new sample frequencies are added (W at a new real frequency is computed for all
    requested orbitals at once).  The fits therefore always have a sample within T1 of every
    residue actually used.
  * The real-axis W uses the same eta-broadened response function as pyscf.gw.gw_cd, so that
    fsCD reproduces canonical CD in the limit T1 -> 0.  Samples at real x are placed at
    z = x + i eta in the fit.
  * Memory: besides the DF integrals Lpq (the only copy; no separate occupied-virtual block
    is kept), the persistent arrays are W^c on the imaginary grid, (nw, norbs, nmo), and the
    real-axis samples, which are stored only for the pairs that can carry a residue (occupied
    p with occupied m, virtual p with virtual m).  The response function is accumulated over
    blocks of occupied orbitals and W over blocks of orbitals p, with block sizes chosen
    from gw.max_memory.
  * Accuracy is controlled by fs_t1 and fs_exact_cutoff.  W varies rapidly at small
    frequencies (|w - e_m| of the order of the valence excitation energies, where its poles
    are dense).  Residues there occur for inner-valence states (e.g. the 2s-like levels of
    first-row molecules) and for the self-residue m = p of core states with large QP shifts.
    With pure frequency sampling at the paper's T1 = 0.05 these can deviate from canonical
    CD by tenths of an eV (inner valence) or tens of meV (core self-energy between samples).
    Either reduce fs_t1 or set fs_exact_cutoff (e.g. 1-2 a.u.) to evaluate those few
    residues exactly; the residues that make core states expensive (the valence states seen
    from a core level, at |w - e_m| of tens of a.u.) are still sampled.
'''

import time
import warnings
import numpy as np
import scipy.linalg as sla
from scipy.optimize import newton, brentq

from pyscf import lib
from pyscf.lib import logger, temporary_env
from pyscf.gw.gw_ac import set_frozen_orbs, _mo_energy_without_core, _mo_without_core
from pyscf.gw.gw_cd import GWCD
from pyscf.gw.utils.ac_grid import _get_scaled_legendre_roots

einsum = lib.einsum


# ------------------------------------------------------------------------------ AAA


def aaa_fit(z, f, rtol=1e-13, max_terms=100):
    '''AAA rational approximation in barycentric form (Nakatsukasa, Sete, Trefethen,
    SIAM J. Sci. Comput. 40, A1494 (2018)).

    Support points are added greedily at the sample with the largest error; the weights solve
    min ||A w|| subject to ||w|| = 1 for the Loewner matrix A of the remaining samples (SVD).

    Non-finite samples and repeated sample points (which would make the Loewner matrix
    infinite) are dropped with a warning before fitting.

    Returns (zj, fj, wj) for r(x) = sum_j wj fj / (x - zj) / sum_j wj / (x - zj).
    '''
    z = np.asarray(z, dtype=complex)
    f = np.asarray(f, dtype=complex)
    keep = np.isfinite(z) & np.isfinite(f)
    _, first = np.unique(z[keep], return_index=True)
    idx = np.flatnonzero(keep)[np.sort(first)]
    if idx.size < z.size:
        warnings.warn('aaa_fit: dropped %d non-finite and %d repeated samples of %d'
                      % (z.size - np.count_nonzero(keep), np.count_nonzero(keep) - idx.size,
                         z.size), RuntimeWarning)
        if idx.size == 0:
            raise ValueError('aaa_fit: no finite samples')
        z, f = z[idx], f[idx]
    M = z.size
    max_terms = min(max_terms, M)
    scale = np.abs(f).max()
    mask = np.ones(M, dtype=bool)          # samples not (yet) used as support points
    support = []
    r = np.full(M, f.mean())
    wj = np.ones(1, dtype=complex)
    if scale == 0:
        return z[:1], f[:1], wj
    for _ in range(max_terms):
        j = int(np.argmax(np.where(mask, np.abs(f - r), -1.0)))
        support.append(j)
        mask[j] = False
        zj, fj = z[support], f[support]
        if not mask.any():
            wj = np.ones(len(support), dtype=complex) / np.sqrt(len(support))
            break
        C = 1.0 / (z[mask, None] - zj[None, :])
        A = (f[mask, None] - fj[None, :]) * C
        # right singular vector of the smallest singular value (full V if A is wide)
        wj = _min_right_singular_vector(A)
        r = f.copy()
        r[mask] = (C @ (wj * fj)) / (C @ wj)
        if np.abs(f - r).max() <= rtol * scale:
            break
    return z[support], f[support], wj


def _min_right_singular_vector(A):
    '''Right singular vector of the smallest singular value of A (full V if A is wide).
    LAPACK's divide-and-conquer SVD (gesdd, numpy's default) occasionally fails to converge
    for ill-conditioned Loewner matrices; fall back to the slower but more robust gesvd.'''
    full = A.shape[0] < A.shape[1]
    try:
        vh = np.linalg.svd(A, full_matrices=full)[2]
    except np.linalg.LinAlgError:
        warnings.warn('aaa_fit: gesdd SVD did not converge for a %dx%d Loewner matrix; '
                      'retrying with gesvd' % A.shape, RuntimeWarning)
        vh = sla.svd(A, full_matrices=full, lapack_driver='gesvd')[2]
    return vh[-1].conj()


def aaa_eval(zj, fj, wj, x, tol=5e-6):
    '''Evaluate the barycentric rational function at scalar x.  Within tol of a support point
    the sampled value is returned.'''
    d = x - zj
    k = int(np.argmin(np.abs(d)))
    if abs(d[k]) < tol:
        return fj[k]
    c = wj / d
    return np.dot(c, fj) / c.sum()


# ------------------------------------------------------------------- real-axis grid


def build_real_grid(freqs, t1=0.05, t2=1e-3):
    '''Sparse real-axis sample frequencies from a list of residue frequencies |w - e_m| >= 0.

    Starting from 0, a sorted candidate z_j is discarded if, relative to the last kept
    point z_i, |z_i - z_j| < t1 or |z_i - z_j| / min(z_i, z_j) < t2 (Eqs. 24-25 of the paper).
    '''
    cand = np.abs(np.asarray(freqs, dtype=float))
    return np.concatenate([[0.0], _merge_new_points(cand, [0.0], t1, t2)])


def _merge_new_points(cand, existing, t1, t2):
    '''Candidates not represented by a point of `existing` or by an earlier kept candidate.
    A point z is represented by a kept point z_i if |z - z_i| < t1 or
    |z - z_i| / min(z, z_i) < t2.  Candidates are processed in ascending order.'''
    kept = list(existing)
    new = []
    for z in np.sort(cand):
        if kept:
            k = np.asarray(kept)
            d = np.abs(k - z)
            j = int(np.argmin(d))
            lo = min(z, kept[j])
            if d[j] < t1 or (lo > 0 and d[j] / lo < t2):
                continue
        kept.append(z)
        new.append(z)
    return np.asarray(new)


def residue_frequencies(ef, omega, mo_energy):
    '''Residues of R^c(omega): orbital indices, weights f_m and frequencies |omega - e_m|.

    f_m = +1 for e_F < e_m < omega, -1 for omega < e_m < e_F, and +-1/2 for e_m = omega
    (Eq. 11 of the paper; needed for consistency with the analytic W(0) term of I^c).
    '''
    if omega > ef:
        idx = np.where((mo_energy <= omega) & (mo_energy > ef))[0]
        fm = np.where(mo_energy[idx] == omega, 0.5, 1.0)
    else:
        idx = np.where((mo_energy >= omega) & (mo_energy < ef))[0]
        fm = np.where(mo_energy[idx] == omega, -0.5, -1.0)
    return idx, fm, np.abs(mo_energy[idx] - omega)


def _dielectric(Lpq, nocc, mo_energy, omega, eta, words):
    '''eps = 1 - Pi in the auxiliary basis, accumulated over blocks of occupied orbitals.

    eta None: Pi(i omega) on the imaginary axis (real; as pyscf.gw.gw_ac.get_rho_response).
    Otherwise: Pi(omega + i eta) on the real axis with the broadening of canonical CD
    (complex symmetric; as pyscf.gw.gw_cd.get_rho_response_R).
    '''
    naux, nmo = Lpq.shape[:2]
    nvir = nmo - nocc
    iblk = max(1, min(nocc, words // max(3 * naux * nvir, 1)))
    er = np.zeros((naux, naux))
    ei = None if eta is None else np.zeros((naux, naux))
    for i0, i1 in lib.prange(0, nocc, iblk):
        Lia = np.ascontiguousarray(Lpq[:, i0:i1, nocc:]).reshape(naux, -1)
        eia = (mo_energy[i0:i1, None] - mo_energy[None, nocc:]).ravel()
        if eta is None:
            er -= (Lia * (4.0 * eia / (omega**2 + eia**2))) @ Lia.T
        else:
            f = 2.0 * (1.0 / (omega + eia + 2j * eta) + 1.0 / (-omega + eia))
            er -= (Lia * f.real) @ Lia.T
            ei -= (Lia * f.imag) @ Lia.T
    er[np.diag_indices(naux)] += 1.0
    if ei is None:
        return er
    eps = er.astype(complex)
    del er
    eps.imag = ei
    return eps


class _WindowLpq:
    '''(naux, nmo, nmo) array-like over the only parts of the MO DF integrals fsCD-GW uses:
    the occupied-virtual block (response function) and the rows of the GW orbitals (window
    rows, all columns; self-energy and W_{pm,pm}).  Used with GWFSCD.lowmem_ints.'''

    ndim = 3
    dtype = np.dtype(np.float64)

    def __init__(self, ov, rows, orbs, nocc, nmo):
        self.ov = ov
        self.rows = rows
        self.orbs = list(orbs)
        self.pos = {p: k for k, p in enumerate(self.orbs)}
        self.nocc = nocc
        self.shape = (ov.shape[0], nmo, nmo)

    @property
    def nbytes(self):
        return self.ov.nbytes + self.rows.nbytes

    def __getitem__(self, key):
        if not (isinstance(key, tuple) and len(key) == 3 and key[0] == slice(None)):
            raise IndexError('_WindowLpq supports Lpq[:, rows, cols] only')
        _, r, c = key
        o, n = self.nocc, self.shape[1]
        if isinstance(r, slice):
            r0, r1, sr = r.indices(n)
            vir_cols = isinstance(c, slice) and c.indices(n) == (o, n, 1)
            if sr == 1 and r1 <= o and vir_cols:
                return self.ov[:, r0:r1]
            rows = list(range(r0, r1, sr))
        else:
            rows = [int(p) for p in np.atleast_1d(r)]
        try:
            idx = [self.pos[p] for p in rows]
        except KeyError:
            raise IndexError('rows %s are not among the GW orbitals of _WindowLpq' % rows)
        if idx == list(range(idx[0], idx[0] + len(idx))):
            return self.rows[:, idx[0]:idx[0] + len(idx), c]
        return self.rows[:, idx][:, :, c]


def _transform_window(gw, mo_coeff, orbs):
    '''_WindowLpq for the GW orbitals orbs (non-frozen indexing), from gw.with_df.'''
    from pyscf.gw.bse_lowmem import transform_df_pairs
    nocc = gw.nocc
    nmo = mo_coeff.shape[1]
    pairs = {'ov': (slice(0, nocc), slice(nocc, nmo)), 'rows': (list(orbs), slice(0, nmo))}
    with temporary_env(gw.with_df, verbose=0), temporary_env(gw.mol, verbose=0):
        blk = transform_df_pairs(gw.with_df, mo_coeff, pairs, gw.max_memory)
    return _WindowLpq(blk['ov'], blk['rows'], orbs, nocc, nmo)


def _df_outcore_if_large(gw, frac=0.3):
    '''Store the AO DF tensor of gw.with_df on disk (lib.param.TMPDIR) if it would take more
    than frac of the memory left under gw.max_memory and has not been built yet.'''
    from pyscf import df
    with_df = gw.with_df
    if with_df._cderi is not None or isinstance(with_df._cderi_to_save, str):
        return
    mol = gw.mol
    naux = df.addons.make_auxmol(mol, with_df.auxbasis).nao_nr()
    nao = mol.nao_nr()
    size_mb = naux * nao * (nao + 1) // 2 * 8 / 1e6
    if size_mb > frac * (gw.max_memory - lib.current_memory()[0]):
        # kept alive on the GW object; the file is removed when it is garbage collected
        gw._cderi_tmpfile = lib.NamedTemporaryFile(dir=lib.param.TMPDIR)
        with_df._cderi_to_save = gw._cderi_tmpfile.name
        logger.info(gw, 'fsCD: AO DF tensor (%.0f MB) stored on disk in %s', size_mb,
                    with_df._cderi_to_save)


def _orbital_block(Lpq, orbs, cols):
    '''Lpq[:, orbs, cols] as a (naux, len(orbs) * ncols) matrix; a view (no copy) when orbs
    is a contiguous range and cols spans all orbitals.'''
    naux = Lpq.shape[0]
    p0 = orbs[0]
    if list(orbs) == list(range(p0, p0 + len(orbs))):
        return np.ascontiguousarray(Lpq[:, p0:p0 + len(orbs), cols]).reshape(naux, -1)
    return np.ascontiguousarray(Lpq[:, orbs, cols]).reshape(naux, -1)


def get_W_diag(Lpq, nocc, mo_energy, freqs, groups, eta=None, max_memory=4000):
    '''Diagonal elements W^c_{pm,pm} = [L (eps^-1 - 1) L]_{pm,pm} of the correlation part of
    the screened interaction.

    freqs : imaginary frequencies (eta None; real result) or real frequencies, evaluated at
        x + i eta with the response function of canonical CD (complex result).
    groups : list of (orbs, cols), orbs a list of orbital indices p and cols a slice of
        orbitals m.  Returns a list with one array (nfreq, len(orbs), ncols) per group.

    eps^-1 - 1 is formed in place once per frequency (no identity matrices; with N^2
    right-hand sides a GEMM with the inverse is faster than triangular solves) and the
    orbitals are processed in blocks, so the temporaries are bounded by max_memory (MB)
    apart from a few naux x naux matrices.
    '''
    naux, nmo = Lpq.shape[:2]
    cplx = eta is not None
    dtype = complex if cplx else float
    freqs = np.asarray(freqs, dtype=float)
    out = [np.empty((freqs.size, len(orbs), len(range(nmo)[cols])), dtype=dtype) for orbs, cols in groups]
    # doubles available for temporaries (half of what is left under max_memory), minus the
    # naux x naux matrices of the response function
    words = int(max(max_memory - lib.current_memory()[0], 100) * 0.5e6 / 8) - 4 * naux * naux
    words = max(words, 1)
    for iw, w in enumerate(freqs):
        eps = _dielectric(Lpq, nocc, mo_energy, w, eta, words)
        # eps is (complex) symmetric: invert its transpose, the Fortran-ordered view, in place
        epsinv = sla.inv(eps.T, overwrite_a=True, check_finite=False)
        del eps
        epsinv[np.diag_indices(naux)] -= 1.0
        for (orbs, cols), res in zip(groups, out):
            ncol = res.shape[2]
            if len(orbs) == 0 or ncol == 0:
                continue
            per_orb = naux * ncol * (3 if cplx else 2)
            kblk = max(1, min(len(orbs), words // max(per_orb, 1)))
            for k0, k1 in lib.prange(0, len(orbs), kblk):
                lblk = _orbital_block(Lpq, orbs[k0:k1], cols)
                Q = epsinv @ lblk
                Q *= lblk
                res[iw, k0:k1] = Q.sum(axis=0).reshape(k1 - k0, ncol)
                del lblk, Q
        del epsinv
    return out


class _RealSamples:
    '''Growing set of real-axis sample frequencies and W^c_{pm,pm} sampled there.

    Only the elements that can enter a residue are stored: for an occupied orbital p the
    residues of R^c (omega < e_F) belong to occupied m, for a virtual p to virtual m.  W is
    kept as a list of per-batch arrays, so adding samples does not copy what is stored.
    Residues of other pairs (only if the QP equation is evaluated on the other side of the
    Fermi level) are evaluated exactly by get_sigma_c.
    '''

    def __init__(self, gw, orbs, Lpq, mo_energy):
        self.gw = gw
        self.Lpq = Lpq
        self.mo_energy = mo_energy
        nocc = gw.nocc
        nmo = Lpq.shape[1]
        self.nocc = nocc
        occ = [k for k, p in enumerate(orbs) if p < nocc]
        vir = [k for k, p in enumerate(orbs) if p >= nocc]
        self.groups = [([orbs[k] for k in occ], slice(0, nocc)), ([orbs[k] for k in vir], slice(nocc, nmo))]
        self.col0 = [0, nocc]
        self.ncol = [nocc, nmo - nocc]
        self.where = {}
        for g, ks in enumerate((occ, vir)):
            for j, k in enumerate(ks):
                self.where[k] = (g, j)
        self.freqs = np.zeros(0)
        self._W = [[], []]

    def add(self, freqs):
        freqs = np.asarray(freqs, dtype=float)
        if freqs.size == 0:
            return 0
        W = get_W_diag(self.Lpq, self.nocc, self.mo_energy, freqs, self.groups, eta=self.gw.eta,
                       max_memory=self.gw.max_memory)
        for g in range(2):
            self._W[g].append(W[g])
        self.freqs = np.concatenate([self.freqs, freqs])
        return freqs.size

    def owns(self, k, m):
        '''Mask: is W_{pm,pm} sampled for orbital k of the GW list and orbitals m?'''
        g, _ = self.where[k]
        c = np.asarray(m) - self.col0[g]
        return (c >= 0) & (c < self.ncol[g])

    def series(self, k, m):
        '''W_{pm,pm} at all sample frequencies (m must be owned).'''
        g, j = self.where[k]
        c = m - self.col0[g]
        return np.concatenate([W[:, j, c] for W in self._W[g]])

    def ensure(self, cand):
        '''Add samples so that every candidate residue frequency is represented.'''
        new = _merge_new_points(cand, self.freqs, self.gw.fs_t1, self.gw.fs_t2)
        if new.size:
            logger.debug(self.gw, 'fsCD: adding %d real sample frequencies %s', new.size, new)
        return self.add(new)


class _WFit:
    '''AAA fits of W^c_{pm,pm}(z) for one orbital p (index k in the GW orbital list), built
    lazily and rebuilt when the real-axis sample set grows.'''

    def __init__(self, k, zI, WI, samples, eta, rtol, max_terms):
        self.k = k
        self.zI = zI
        self.WI = WI              # (nw, nmo) real
        self.samples = samples
        self.eta = eta
        self.rtol = rtol
        self.max_terms = max_terms
        self.fits = {}
        self.nreal = -1

    def __call__(self, m, x):
        '''W^c_{pm,pm}(x + i eta) for real x >= 0 (m must be owned by the samples).'''
        s = self.samples
        if s.freqs.size != self.nreal:
            self.fits = {}
            self.nreal = s.freqs.size
            self.zs = np.concatenate([self.zI, s.freqs + 1j * self.eta])
        fit = self.fits.get(m)
        if fit is None:
            f = np.concatenate([self.WI[:, m], s.series(self.k, m)])
            fit = self.fits[m] = aaa_fit(self.zs, f, self.rtol, self.max_terms)
        return aaa_eval(*fit, x + 1j * self.eta)


# ------------------------------------------------------------------------------ kernel


def prepare(gw):
    '''Energy-independent quantities, computed once per object: frozen-orbital bookkeeping, DF
    integrals in the MO basis, exchange self-energy and mean-field vxc.'''
    if gw._fs_ints is not None:
        return gw._fs_ints
    mf = gw._scf
    nocc = gw.nocc
    set_frozen_orbs(gw)
    mo_coeff_frz = _mo_without_core(gw, gw.mo_coeff)
    if gw.Lpq is None and gw.lowmem_ints:
        # only the occupied-virtual block and the rows of the GW orbitals
        _df_outcore_if_large(gw)
        Lpq = _transform_window(gw, mo_coeff_frz, gw.orbs_frz)
        logger.info(gw, 'fsCD: low-memory MO integrals (ov block + %d orbital rows), %.0f MB',
                    len(gw.orbs_frz), Lpq.nbytes / 1e6)
    else:
        if gw.Lpq is None:
            with temporary_env(gw.with_df, verbose=0), temporary_env(gw.mol, verbose=0):
                gw.Lpq = gw.ao2mo(mo_coeff_frz)
        Lpq = gw.Lpq
    with temporary_env(mf, verbose=0):
        gw.vxc = mo_coeff_frz.T @ (mf.get_veff() - mf.get_j()) @ mo_coeff_frz
    if gw.vhf_df and isinstance(Lpq, _WindowLpq):
        # only the GW-orbital block of Sigma^x is needed (its diagonal enters the QP equation)
        nmo = Lpq.shape[1]
        rows_occ = Lpq.rows[:, :, :nocc]
        win = np.asarray(gw.orbs_frz)
        gw.vk = np.zeros((nmo, nmo))
        gw.vk[win[:, None], win] = -einsum('Lpi,Lqi->pq', rows_occ, rows_occ)
    elif gw.vhf_df:
        gw.vk = -einsum('Lpi,Liq->pq', Lpq[:, :, :nocc], Lpq[:, :nocc, :])
    else:
        gw.vk = gw.get_sigma_exchange(mo_coeff=mo_coeff_frz)
    gw._fs_ints = {'orbs_frz': list(gw.orbs_frz), 'Lpq': Lpq,
                   'mo_energy_mf': _mo_energy_without_core(gw, mf.mo_energy)}
    return gw._fs_ints


def setup(gw, mo_energy_g=None, mo_energy_w=None, ef=None):
    '''Build the self-energy machinery for given orbital energies (non-frozen space):
    imaginary-axis W^c, the initial real-axis sample grid and (empty) AAA fit caches.

    mo_energy_g : energies in the Green's function (residue positions, imaginary-axis G)
    mo_energy_w : energies in the screening (response function of W)
    ef : Fermi level
    All default to the mean-field values (G0W0).
    '''
    log = logger.new_logger(gw)
    ints = prepare(gw)
    orbs_frz, Lpq = ints['orbs_frz'], ints['Lpq']
    nocc, nmo = gw.nocc, Lpq.shape[1]
    e_g = ints['mo_energy_mf'] if mo_energy_g is None else np.asarray(mo_energy_g, dtype=float)
    e_w = ints['mo_energy_mf'] if mo_energy_w is None else np.asarray(mo_energy_w, dtype=float)
    if ef is None:
        ef = gw.get_ef(mo_energy=gw._scf.mo_energy)
    gw.ef = ef

    # imaginary axis: quadrature and W^c(i w), shared by I^c and the fits
    quad_freqs, quad_wts = _get_scaled_legendre_roots(gw.nw)
    # (W^c(i0) is the last frequency)
    log.info('fsCD: computing W on %d imaginary frequencies', gw.nw)
    WmnI = get_W_diag(Lpq, nocc, e_w, np.append(quad_freqs, 0.0), [(orbs_frz, slice(0, nmo))],
                      max_memory=gw.max_memory)[0]
    WmnI0 = WmnI[-1]
    WmnI = WmnI[:-1]

    # real axis: initial sparse grid from the residues at the current energies (plus 0);
    # extended on the fly whenever a residue at the current frequency is not represented
    samples = _RealSamples(gw, orbs_frz, Lpq, e_w)
    cand = [residue_frequencies(ef, e_g[p], e_g)[2] for p in orbs_frz]
    samples.add(build_real_grid(np.concatenate(cand), gw.fs_t1, gw.fs_t2))
    log.info('fsCD: %d residue frequencies -> %d initial real sample frequencies',
             sum(len(c) for c in cand), samples.freqs.size)

    fits = [_WFit(k, 1j * quad_freqs, WmnI[:, k], samples, gw.eta, gw.fs_aaa_rtol, gw.fs_aaa_max_terms)
            for k in range(len(orbs_frz))]
    gw._fs = {'mo_energy': e_g, 'mo_energy_w': e_w, 'quad': (quad_freqs, quad_wts), 'WmnI': WmnI,
              'WmnI0': WmnI0,
              'samples': samples, 'fits': fits, 'sign': np.sign(ef - e_g),
              'orbs_frz': orbs_frz, 'Lpq': Lpq}
    return gw._fs


def get_sigma_c(gw, k, omega):
    '''Correlation self-energy Sigma^c_pp(omega) for the k-th orbital of gw.orbs.'''
    fs = gw._fs
    e = fs['mo_energy']
    quad_freqs, quad_wts = fs['quad']
    # I^c: imaginary-axis integral.  With a_m = omega - e_m, the W^c(i0) part is integrated
    # analytically, int_0^inf a / (a^2 + nu^2) dnu = (pi/2) sign(a), and only the remainder
    # W^c(i nu) - W^c(i0) by quadrature.  This is exact in the continuum and keeps I^c + R^c
    # smooth when omega crosses a pole e_m of G (where R^c jumps by W^c(0)); a quadrature of
    # the full integrand cannot resolve this step for |a| below its smallest frequencies.
    # evGW converges onto exactly this situation (omega -> e_p of the previous cycle).
    emo = omega - 1j * gw.eta * fs['sign'] - e
    W0 = fs['WmnI0'][k]
    g0 = quad_wts[None, :] * emo[:, None] / ((emo**2)[:, None] + (quad_freqs**2)[None, :])
    sigma = -einsum('mw,wm', g0, fs['WmnI'][:, k] - W0[None, :]) / np.pi
    sigma -= 0.5 * np.dot(np.sign(omega - e), W0)
    # R^c: residues from the rational fits of W^c; optionally the few residues at small
    # frequencies (among the valence excitation poles of W) exactly, as in canonical CD.
    # Residues of pairs (p, m) that are not sampled (omega on the other side of the Fermi
    # level than e_p, e.g. a wandering Newton iterate) are evaluated exactly as well.
    idx, fm, zm = residue_frequencies(gw.ef, omega, e)
    cutoff = gw.fs_exact_cutoff or 0.0
    samples = fs['samples']
    exact = (zm < cutoff) | ~samples.owns(k, idx)
    if exact.any():
        p = fs['orbs_frz'][k]
        Wex = get_W_diag(fs['Lpq'], gw.nocc, fs['mo_energy_w'], zm[exact],
                         [([p], slice(0, e.size))], eta=gw.eta, max_memory=gw.max_memory)[0][:, 0]
        sigma += np.dot(fm[exact], Wex[np.arange(Wex.shape[0]), idx[exact]])
    idx, fm, zm = idx[~exact], fm[~exact], zm[~exact]
    samples.ensure(zm)
    wfit = fs['fits'][k]
    for m, f, x in zip(idx, fm, zm):
        sigma += f * wfit(m, x)
    return sigma


def solve_qp(gw, guess=None):
    '''Solve omega = e^MF_p + Sigma^c_pp(omega) + Sigma^x_pp - vxc_pp for all orbitals in
    gw.orbs with the self-energy set up by setup().  guess: starting energies (full MO space).

    Returns (conv, conv_inds, mo_energy) in the full MO space (zero outside gw.orbs).'''
    mf = gw._scf
    nocc = gw.nocc
    log = logger.new_logger(gw)
    vk, v_mf = gw.vk, gw.vxc
    mo_energy = np.zeros_like(mf.mo_energy)
    conv_inds = np.zeros(mo_energy.shape, dtype=int)
    conv = True
    for k, (p_frz, p_all) in enumerate(zip(gw.orbs_frz, gw.orbs)):

        def quasiparticle(omega):
            sigma = get_sigma_c(gw, k, omega).real
            return omega - mf.mo_energy[p_all] - (sigma + vk[p_frz, p_frz] - v_mf[p_frz, p_frz])

        if guess is None:
            x0 = mf.mo_energy[p_all] + (-1e-2 if p_frz < nocc else 1e-2)
        else:
            x0 = guess[p_all]
        e = None
        try:
            e, res = newton(quasiparticle, x0, tol=gw.qpe_tol, maxiter=gw.qpe_max_iter, full_output=True)
            log.debug('QP energy for orb %d: %.10f (%d iterations)', p_all, e, res.iterations)
        except RuntimeError:
            e = _qp_bracket(quasiparticle, x0, gw.qpe_scan_range, gw.qpe_scan_step, gw.qpe_tol)
            if e is not None:
                log.debug('QP energy for orb %d: %.10f (bracketing fallback)', p_all, e)
        if e is not None and gw.fs_polish:
            e = _polish(gw, quasiparticle, e, log, p_all, k)
        if e is None:
            conv = False
            log.warn('QP equation for orbital %d did not converge', p_all)
        else:
            mo_energy[p_all] = e
            conv_inds[p_all] = 1
    return conv, conv_inds, mo_energy


def _polish(gw, f, e, log, p_all, k, max_rounds=4):
    '''Make the root exact: sample W at every fitted residue of the solution that is not yet
    a sample frequency (within 5e-6 a.u.) and restart Newton from there, until the residues
    of the root are all sample points.  The result then solves the canonical-CD QP equation.'''
    fs = gw._fs
    samples = fs['samples']
    cutoff = gw.fs_exact_cutoff or 0.0
    for _ in range(max_rounds):
        idx, _, zm = residue_frequencies(gw.ef, e, fs['mo_energy'])
        zm = zm[(zm >= cutoff) & samples.owns(k, idx)]
        if samples.freqs.size:
            dist = np.abs(zm[:, None] - samples.freqs[None, :]).min(axis=1)
            zm = zm[dist > 5e-6]
        if zm.size == 0:
            return e
        samples.add(np.unique(zm))
        try:
            e = newton(f, e, tol=gw.qpe_tol, maxiter=gw.qpe_max_iter)
        except RuntimeError:
            log.warn('polishing the QP solution of orbital %d failed', p_all)
            return None
    return e


def _qp_bracket(f, x0, width, step, tol):
    '''Fallback QP solver: scan [x0 - width, x0 + width] for sign changes of
    f(w) = w - e - Re Sigma(w) from negative to positive (genuine solutions; the opposite
    crossings are the broadened poles of Sigma) and refine the one closest to x0 with Brent's
    method.  Returns None if there is none.'''
    xs = x0 + np.arange(-width, width + 0.5 * step, step)
    fs = np.array([f(x) for x in xs])
    up = np.where((fs[:-1] < 0) & (fs[1:] >= 0))[0]
    if up.size == 0:
        return None
    i = up[np.argmin(np.abs(0.5 * (xs[up] + xs[up + 1]) - x0))]
    return brentq(f, xs[i], xs[i + 1], xtol=tol)


def kernel(gw):
    '''fsCD-G0W0 kernel.  Returns (conv, conv_inds, mo_energy) like pyscf.gw.gw_cd.kernel.'''
    setup(gw)
    conv, conv_inds, mo_energy = solve_qp(gw)
    gw.real_freqs = gw._fs['samples'].freqs
    logger.info(gw, 'fsCD: %d real sample frequencies in total', gw.real_freqs.size)
    return conv, conv_inds, mo_energy


class GWFSCD(GWCD):
    '''Frequency-sampled contour deformation G0W0 (fsCD-GW).

    Attributes (in addition to GWCD):
        fs_t1 : float
            Absolute merge threshold of the real-axis grid (a.u.), default 0.05 (paper value).
            Residues closer than this to a sample frequency are taken from the rational fit;
            this is the main accuracy control.
        fs_t2 : float
            Relative merge threshold of the real-axis grid, default 1e-3.
        fs_aaa_rtol : float
            Relative tolerance of the AAA fits, default 1e-13.
        fs_aaa_max_terms : int
            Maximum number of AAA support points, default 100.
        fs_exact_cutoff : float or None
            Residues with frequency |w - e_m| below this value (a.u.) are evaluated exactly
            (one dielectric matrix each, as in canonical CD) instead of from the fit.  These
            lie among the valence excitation poles of W, where W varies too rapidly to be
            interpolated from samples T1 apart: the self-residue m = p of a core state with a
            large QP shift and its near-degenerate partners, and the residues of inner-valence
            states.  Their number does not grow with system size for core states, so the
            O(N^4) scaling is kept.  Default None (pure frequency sampling, as in the paper);
            1-2 a.u. makes fsCD agree with canonical CD to about a meV for all states.

        fs_polish : bool
            After the QP equation is solved, sample W exactly at the residues of the solution
            that are not yet sample frequencies and re-solve, so that the quasiparticle energy
            solves the canonical-CD equation (fitted values are then only used along the way).
            Costs at most about one canonical-CD self-energy evaluation per orbital.  Default
            False (pure frequency sampling); EVGWFSCD enables it.
        lowmem_ints : bool
            Do not form the full MO DF tensor Lpq (naux, nmo, nmo).  Only the occupied-virtual
            block (response function) and the rows of the orbitals in gw.orbs (self-energy) are
            transformed, and the AO DF tensor is written to disk (lib.param.TMPDIR) when it
            would take more than 30% of the memory left under max_memory.  gw.Lpq stays None.
            Same results; requires that no other code needs gw.Lpq afterwards (QRBSE builds its
            own integral blocks from gw.with_df).  Default False.
        qpe_scan_range, qpe_scan_step : float
            If Newton's method fails for the QP equation, the solution closest to the starting
            guess is bracketed on a grid of this half-width and step (a.u., default 0.5 and
            0.005) and refined with Brent's method.

    Saved results (in addition to GWCD):
        real_freqs : real-axis sample frequencies used.
    '''

    def __init__(self, mf, frozen=None, auxbasis=None):
        super().__init__(mf, frozen=frozen, auxbasis=auxbasis)
        self.fs_t1 = 0.05
        self.fs_t2 = 1e-3
        self.fs_aaa_rtol = 1e-13
        self.fs_aaa_max_terms = 100
        self.fs_exact_cutoff = None
        self.qpe_scan_range = 0.5     # fallback root bracketing: +-range around the guess (Ha)
        self.qpe_scan_step = 0.005    # and step of the scan (Ha)
        self.fs_polish = False
        self.lowmem_ints = False
        self.real_freqs = None
        self._fs = None
        self._fs_ints = None

    def dump_flags(self):
        super().dump_flags()
        log = logger.new_logger(self)
        log.info('fsCD real-grid thresholds T1 = %.3g a.u., T2 = %.3g', self.fs_t1, self.fs_t2)
        log.info('fsCD AAA rtol = %.1e, max terms = %d', self.fs_aaa_rtol, self.fs_aaa_max_terms)
        log.info('fsCD exact residues below %s a.u., polish = %s', self.fs_exact_cutoff, self.fs_polish)

    def _check_input(self):
        if self.qpe_linearized:
            raise NotImplementedError('fsCD-GW solves the full quasiparticle equation')
        if self.outcore:
            raise NotImplementedError('fsCD-GW does not support outcore')
        if self.Lpq is None:
            self.initialize_df(auxbasis=self.auxbasis)

    def kernel(self):
        '''Do one-shot G0W0 with frequency-sampled contour deformation.'''
        self._check_input()
        cput0 = (time.process_time(), time.perf_counter())
        self.dump_flags()
        self.converged, self.conv_inds, self.mo_energy = kernel(self)
        logger.timer(self, 'GWFSCD', *cput0)
        return self.mo_energy

    def get_sigma_c(self, p, omega):
        '''Correlation self-energy Sigma^c_pp(omega) (complex, a.u.) of orbital p (index in the
        full MO space, must be in self.orbs) at one or several real frequencies.'''
        if self._fs is None:
            self._check_input()
            setup(self)
        k = list(self.orbs).index(p)
        omega = np.asarray(omega, dtype=float)
        out = np.array([get_sigma_c(self, k, w) for w in omega.ravel()])
        return out.reshape(omega.shape)
