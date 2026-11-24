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
import numpy as np
import scipy.linalg as sla
from scipy.optimize import newton, brentq

from pyscf import lib
from pyscf.lib import logger, temporary_env
from pyscf.gw.gw_ac import set_frozen_orbs, _mo_energy_without_core, _mo_without_core
from pyscf.gw.gw_cd import GWCD, get_WmnI_diag, get_rho_response_R
from pyscf.gw.utils.ac_grid import _get_scaled_legendre_roots
from pyscf.gw.utils.gw_np_helper import mkslice

einsum = lib.einsum


# ------------------------------------------------------------------------------ AAA


def aaa_fit(z, f, rtol=1e-13, max_terms=100):
    '''AAA rational approximation in barycentric form (Nakatsukasa, Sete, Trefethen,
    SIAM J. Sci. Comput. 40, A1494 (2018)).

    Support points are added greedily at the sample with the largest error; the weights solve
    min ||A w|| subject to ||w|| = 1 for the Loewner matrix A of the remaining samples (SVD).

    Returns (zj, fj, wj) for r(x) = sum_j wj fj / (x - zj) / sum_j wj / (x - zj).
    '''
    z = np.asarray(z, dtype=complex)
    f = np.asarray(f, dtype=complex)
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
        wj = np.linalg.svd(A, full_matrices=A.shape[0] < A.shape[1])[2][-1].conj()
        r = f.copy()
        r[mask] = (C @ (wj * fj)) / (C @ wj)
        if np.abs(f - r).max() <= rtol * scale:
            break
    return z[support], f[support], wj


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


def get_WmnR_diag(orbs, Lpq, Lia, real_freqs, mo_energy, eta):
    '''W^c_{pm,pm} on the real-axis sample frequencies, shape (nreal, norbs, nmo), complex.

    Uses the same (eta-broadened) response function as canonical CD (get_rho_response_R).
    '''
    naux, nmo, _ = Lpq.shape
    nocc = Lia.shape[1]
    norbs = len(orbs)
    eia = mo_energy[:nocc, None] - mo_energy[None, nocc:]
    l_slice = Lpq[:, mkslice(orbs), :].reshape(naux, -1)
    eye = np.eye(naux)
    Wmn = np.empty((len(real_freqs), norbs, nmo), dtype=complex)
    for k, w in enumerate(real_freqs):
        Pi = get_rho_response_R(eia, w, Lia, eta)
        # Pi_inv = (I - Pi)^-1 - I
        Pi_inv = sla.solve(eye - Pi, eye, check_finite=False, overwrite_a=True) - eye
        Wmn[k] = np.einsum('Pn,Pn->n', l_slice, Pi_inv @ l_slice).reshape(norbs, nmo)
    return Wmn


class _RealSamples:
    '''Growing set of real-axis sample frequencies and the W^c_{pm,pm} sampled there.'''

    def __init__(self, gw, orbs, Lpq, Lia, mo_energy):
        self.gw = gw
        self.args = (orbs, Lpq, Lia)
        self.mo_energy = mo_energy
        self.freqs = np.zeros(0)
        self.W = np.zeros((0, len(orbs), Lpq.shape[1]), dtype=complex)

    def add(self, freqs):
        freqs = np.asarray(freqs, dtype=float)
        if freqs.size == 0:
            return 0
        orbs, Lpq, Lia = self.args
        W = get_WmnR_diag(orbs, Lpq, Lia, freqs, self.mo_energy, self.gw.eta)
        self.freqs = np.concatenate([self.freqs, freqs])
        self.W = np.concatenate([self.W, W])
        return freqs.size

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
        '''W^c_{pm,pm}(x + i eta) for real x >= 0.'''
        s = self.samples
        if s.freqs.size != self.nreal:
            self.fits = {}
            self.nreal = s.freqs.size
            self.zs = np.concatenate([self.zI, s.freqs + 1j * self.eta])
        fit = self.fits.get(m)
        if fit is None:
            f = np.concatenate([self.WI[:, m], s.W[:, self.k, m]])
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
    if gw.Lpq is None:
        with temporary_env(gw.with_df, verbose=0), temporary_env(gw.mol, verbose=0):
            gw.Lpq = gw.ao2mo(mo_coeff_frz)
    Lpq = gw.Lpq
    with temporary_env(mf, verbose=0):
        gw.vxc = mo_coeff_frz.T @ (mf.get_veff() - mf.get_j()) @ mo_coeff_frz
    if gw.vhf_df:
        gw.vk = -einsum('Lpi,Liq->pq', Lpq[:, :, :nocc], Lpq[:, :nocc, :])
    else:
        gw.vk = gw.get_sigma_exchange(mo_coeff=mo_coeff_frz)
    gw._fs_ints = {'orbs_frz': list(gw.orbs_frz), 'Lpq': Lpq,
                   'Lia': np.ascontiguousarray(Lpq[:, :nocc, nocc:]),
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
    orbs_frz, Lpq, Lia = ints['orbs_frz'], ints['Lpq'], ints['Lia']
    e_g = ints['mo_energy_mf'] if mo_energy_g is None else np.asarray(mo_energy_g, dtype=float)
    e_w = ints['mo_energy_mf'] if mo_energy_w is None else np.asarray(mo_energy_w, dtype=float)
    if ef is None:
        ef = gw.get_ef(mo_energy=gw._scf.mo_energy)
    gw.ef = ef

    # imaginary axis: quadrature and W^c(i w), shared by I^c and the fits
    quad_freqs, quad_wts = _get_scaled_legendre_roots(gw.nw)
    WmnI = get_WmnI_diag(gw, orbs_frz, Lpq, Lia, quad_freqs, e_w)
    WmnI0 = get_WmnI_diag(gw, orbs_frz, Lpq, Lia, np.zeros(1), e_w)[0]   # W^c(i0)

    # real axis: initial sparse grid from the residues at the current energies (plus 0);
    # extended on the fly whenever a residue at the current frequency is not represented
    samples = _RealSamples(gw, orbs_frz, Lpq, Lia, e_w)
    cand = [residue_frequencies(ef, e_g[p], e_g)[2] for p in orbs_frz]
    samples.add(build_real_grid(np.concatenate(cand), gw.fs_t1, gw.fs_t2))
    log.info('fsCD: %d residue frequencies -> %d initial real sample frequencies',
             sum(len(c) for c in cand), samples.freqs.size)

    fits = [_WFit(k, 1j * quad_freqs, WmnI[:, k], samples, gw.eta, gw.fs_aaa_rtol, gw.fs_aaa_max_terms)
            for k in range(len(orbs_frz))]
    gw._fs = {'mo_energy': e_g, 'mo_energy_w': e_w, 'quad': (quad_freqs, quad_wts), 'WmnI': WmnI,
              'WmnI0': WmnI0,
              'samples': samples, 'fits': fits, 'sign': np.sign(ef - e_g),
              'orbs_frz': orbs_frz, 'Lpq': Lpq, 'Lia': Lia}
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
    # frequencies (among the valence excitation poles of W) exactly, as in canonical CD
    idx, fm, zm = residue_frequencies(gw.ef, omega, e)
    cutoff = gw.fs_exact_cutoff or 0.0
    exact = zm < cutoff
    if exact.any():
        p = fs['orbs_frz'][k]
        Wex = get_WmnR_diag([p], fs['Lpq'], fs['Lia'], zm[exact], fs['mo_energy_w'], gw.eta)[:, 0]
        sigma += np.dot(fm[exact], Wex[np.arange(Wex.shape[0]), idx[exact]])
    idx, fm, zm = idx[~exact], fm[~exact], zm[~exact]
    fs['samples'].ensure(zm)
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
            e = _polish(gw, quasiparticle, e, log, p_all)
        if e is None:
            conv = False
            log.warn('QP equation for orbital %d did not converge', p_all)
        else:
            mo_energy[p_all] = e
            conv_inds[p_all] = 1
    return conv, conv_inds, mo_energy


def _polish(gw, f, e, log, p_all, max_rounds=4):
    '''Make the root exact: sample W at every fitted residue of the solution that is not yet
    a sample frequency (within 5e-6 a.u.) and restart Newton from there, until the residues
    of the root are all sample points.  The result then solves the canonical-CD QP equation.'''
    fs = gw._fs
    samples = fs['samples']
    cutoff = gw.fs_exact_cutoff or 0.0
    for _ in range(max_rounds):
        _, _, zm = residue_frequencies(gw.ef, e, fs['mo_energy'])
        zm = zm[zm >= cutoff]
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
