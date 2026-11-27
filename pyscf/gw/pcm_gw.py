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
Non-equilibrium polarizable continuum model (PCM) for GW and BSE.

References:
    I. Duchemin, D. Jacquemin, X. Blase, J. Chem. Phys. 144, 164106 (2016)   (GW + PCM)
    I. Duchemin, C. A. Guido, D. Jacquemin, X. Blase, Chem. Sci. 9, 4430 (2018)   (BSE + PCM)

The slow (nuclear + electronic, eps) solvent response is that of the ground state: it is
contained in the orbitals and orbital energies of the reference mean-field calculation with
PCM (e.g. dft.RKS(mol).PCM()).  The fast (electronic, eps_optical) response screens the
electronic excitations: the bare Coulomb interaction entering W is renormalized by the
reaction field,

    v~ = v + v chi_PCM(eps_optical) v,

which, in the density-fitting representation L = c^-1 (P|pq) (c c^T = (P|Q)), is an
auxiliary-space metric

    C = 1 + X M X^T,    X = c^-1 (P|g),    M = (K^-1 R + R^T K^-T) / 2,

with g the PCM surface charges and K, R the PCM matrices (pyscf.solvent.pcm).  With C = U U^T
the dressed integrals L~ = U^T L give (pq|v~|rs) = L~^T L~, so that the GW and BSE equations
are unchanged when L is replaced by L~ (response function, screened interaction, exchange
self-energy and the exchange term of the BSE kernel).

Two GW schemes (GW paper, section II.C):
    'gw'     : GW with v~ (Delta-GW of the paper).
    'cohsex' : GW with v on the PCM(eps) reference, plus the static COHSEX correction
               Delta_n = <n| Sigma^COHSEX[v~] - Sigma^COHSEX[v] |n> built from the mean-field
               spectrum (GW + Delta-COHSEX, the scheme recommended in the paper).
For BSE, the screened and exchange kernels are always built with v~ (BSE paper, eq. 17).

Not included: the modified IEF-PCM treatment of the charge spilling out of the cavity
(appendix of the GW paper), a frequency-dependent dielectric response and orbital
relaxation.

Supported drivers (all restricted): GWACLowMem, GWCDLowMem, GWFSCD, EVGWLowMem, EVGWFSCD;
BSELowMem and BSEQuadraticResponse through make_bse.
'''

import numpy as np
import scipy.linalg

from pyscf import lib, gto
from pyscf.lib import logger, temporary_env
from pyscf.df import addons as df_addons
from pyscf.df.incore import aux_e2, _eig_decompose, LINEAR_DEP_THR
from pyscf.gw.gw_ac import set_frozen_orbs, _mo_energy_without_core, _mo_without_core
from pyscf.gw.bse_lowmem import transform_df_pairs


# --------------------------------------------------------------- reaction-field metric

def _build_df(with_df):
    if with_df._cderi is None:
        with temporary_env(with_df, verbose=0):
            with_df.build()
    return with_df


def _auxmol(with_df):
    auxmol = getattr(with_df, 'auxmol', None)
    if auxmol is None:
        auxmol = df_addons.make_auxmol(with_df.mol, with_df.auxbasis)
    return auxmol


def _df_frame(with_df, auxmol):
    '''Function A -> F A such that the DF tensor of with_df is F (P|mu nu).

    pyscf builds F from the Cholesky factor of the Coulomb metric (P|Q), or from its
    eigendecomposition when it is linearly dependent; the candidate that reproduces the first
    columns of the stored tensor is returned.
    '''
    mol = with_df.mol
    j2c = auxmol.intor('int2c2e', hermi=1)
    ints = aux_e2(mol, auxmol, intor='int3c2e', aosym='s2ij',
                  shls_slice=(0, 1, 0, 1, 0, auxmol.nbas))
    ints = np.asarray(ints).reshape(-1, auxmol.nao_nr()).T
    ncol = ints.shape[1]
    ref = np.vstack([np.asarray(blk[:, :ncol]) for blk in with_df.loop()])

    cands = []
    try:
        low = scipy.linalg.cholesky(j2c, lower=True)
        cands.append(lambda a: scipy.linalg.solve_triangular(low, a, lower=True))
    except scipy.linalg.LinAlgError:
        pass
    lindep = getattr(with_df, 'linear_dep_threshold', LINEAR_DEP_THR)
    low_eig = _eig_decompose(mol, j2c, lindep)
    cands.append(lambda a: low_eig @ a)

    scale = max(1.0, abs(ref).max())
    for f in cands:
        test = f(ints)
        if test.shape == ref.shape and abs(test - ref).max() < 1e-8 * scale:
            return f
    raise RuntimeError('Cannot identify the metric decomposition of the density-fitting tensor')


def _fast_pcm(pcmobj, eps_fast=None):
    '''Copy of the PCM object with the dielectric constant of the fast response.'''
    from pyscf.solvent.pcm import PCM
    if not isinstance(pcmobj, PCM):
        raise NotImplementedError('GW/BSE solvation is implemented for PCM solvent objects only, got %s'
                                  % type(pcmobj))
    if eps_fast is None:
        eps_fast = pcmobj.get_eps_optical()
    pcm = pcmobj.copy()
    pcm.reset()
    pcm.eps = eps_fast
    pcm.build()
    return pcm


def get_reaction_metric(with_df, pcmobj, eps_fast=None):
    '''Auxiliary-space factor U of the reaction-field-dressed Coulomb metric.

    Args:
        with_df : density-fitting object (built if necessary)
        pcmobj : PCM object (pyscf.solvent.pcm.PCM) defining the cavity and the model
        eps_fast : dielectric constant of the fast response (default pcmobj.get_eps_optical())

    Returns:
        U : (naux, naux) array with U U^T = 1 + X M X^T (see the module docstring), so that
            U^T L is the DF tensor of v + v chi_PCM v
        info : dict with the PCM copy ('pcm'), the reaction part X M X^T ('reac') and the
            smallest eigenvalue of the dressed metric ('min_eig')
    '''
    log = logger.new_logger(pcmobj)
    _build_df(with_df)
    auxmol = _auxmol(with_df)
    pcm = _fast_pcm(pcmobj, eps_fast)
    frame = _df_frame(with_df, auxmol)

    surface = pcm.surface
    fakemol = gto.fakemol_for_charges(surface['grid_coords'], expnt=surface['charge_exp']**2)
    Bt = gto.mole.intor_cross('int2c2e', auxmol, fakemol)        # (P|g)
    X = frame(Bt)                                                  # (naux, ngrids)
    K = pcm._intermediates['K']
    R = pcm._intermediates['R']
    A = np.linalg.solve(K, R)
    M = 0.5 * (A + A.T)
    reac = X @ M @ X.T
    reac = 0.5 * (reac + reac.T)

    C = reac + np.eye(reac.shape[0])
    min_eig = scipy.linalg.eigvalsh(C)[0]
    log.info('GW-PCM: eps_fast = %.4f, %d surface points, smallest eigenvalue of the dressed '
             'metric = %.4e', pcm.eps, X.shape[1], min_eig)
    if min_eig <= 0:
        raise RuntimeError('The reaction-field-dressed Coulomb metric is not positive definite '
                           '(smallest eigenvalue %.3e)' % min_eig)
    U = scipy.linalg.cholesky(C, lower=True)
    return U, {'pcm': pcm, 'reac': reac, 'min_eig': min_eig}


def dress_lpq(Lpq, U, out=None):
    '''L~ = U^T L for a (naux, n1, n2) DF tensor (blocked over the last two indices).'''
    Lpq = np.asarray(Lpq)
    naux = Lpq.shape[0]
    L2 = Lpq.reshape(naux, -1)
    if out is None:
        out = np.empty_like(Lpq)
    o2 = out.reshape(naux, -1)
    blk = max(1, int(2e7 // max(naux, 1)))
    for p0, p1 in lib.prange(0, L2.shape[1], blk):
        o2[:, p0:p1] = U.T @ L2[:, p0:p1]
    return out


def make_dressed_df(with_df, U, max_memory=None):
    '''Density-fitting object whose tensor is U^T times that of with_df.

    The dressed tensor is kept in memory if it takes less than half of the memory left under
    max_memory, otherwise it is written to a temporary HDF5 file (lib.param.TMPDIR).
    '''
    from pyscf import df
    _build_df(with_df)
    mol = with_df.mol
    if max_memory is None:
        max_memory = with_df.max_memory
    naux = with_df.get_naoaux()
    nao = mol.nao_nr()
    npair = nao * (nao + 1) // 2
    dressed = df.DF(mol, auxbasis=with_df.auxbasis)
    dressed.verbose = with_df.verbose
    dressed.stdout = with_df.stdout
    dressed.max_memory = max_memory
    size_mb = naux * npair * 8 / 1e6
    avail = max_memory - lib.current_memory()[0]
    blksize = max(1, int(max(avail, 100) * 0.25e6 / 8 // max(npair, 1)))
    if size_mb < 0.5 * avail:
        out = np.zeros((naux, npair))
        p0 = 0
        for blk in with_df.loop(blksize=blksize):
            p1 = p0 + blk.shape[0]
            out += U[p0:p1].T @ blk
            p0 = p1
        dressed._cderi = out
    else:
        dressed._cderi_tmpfile = lib.H5TmpFile()
        fname = dressed._cderi_tmpfile.filename
        dset = dressed._cderi_tmpfile.create_dataset(dressed._dataname, (naux, npair), 'f8')
        ncol = max(1, int(max(avail, 100) * 0.25e6 / 8 // max(naux, 1)))
        logger.info(with_df, 'GW-PCM: dressed DF tensor (%.0f MB) stored in %s', size_mb, fname)
        for c0, c1 in lib.prange(0, npair, ncol):
            buf = np.zeros((naux, c1 - c0))
            p0 = 0
            for blk in with_df.loop(blksize=blksize):
                p1 = p0 + blk.shape[0]
                buf += U[p0:p1].T @ blk[:, c0:c1]
                p0 = p1
            dset[:, c0:c1] = buf
        dressed._cderi_tmpfile.flush()
        dressed._cderi = fname
    return dressed


# --------------------------------------------------------------- static COHSEX correction

def cohsex_shift(Lov, Lrows, mo_energy, nocc, U):
    '''Static COHSEX correction Delta_n = <n|Sigma^COHSEX[v~] - Sigma^COHSEX[v]|n>.

    Args:
        Lov : (naux, nocc, nvir) occupied-virtual block of the bare DF integrals
        Lrows : (naux, norbs, nmo) bare DF integrals L[:, n, m] of the orbitals n
        mo_energy : mean-field orbital energies (nmo)
        nocc : number of occupied orbitals
        U : dressed metric factor from get_reaction_metric

    Returns:
        (norbs,) array of shifts.  Sigma^SEX_nn = -sum_i (ni|W(0)|in) and
        Sigma^COH_nn = 1/2 sum_m (nm|W(0) - v|mn) (sum over the MO basis), with the static
        RPA screened interaction W(0) built with v or v~.
    '''
    naux = Lov.shape[0]
    eo = mo_energy[:nocc]
    ev = mo_energy[nocc:]
    Pi = np.zeros((naux, naux))
    words = max(int(2e7 // max(naux * Lov.shape[2], 1)), 1)
    for i0, i1 in lib.prange(0, nocc, words):
        L = np.ascontiguousarray(Lov[:, i0:i1]).reshape(naux, -1)
        f = (4.0 / (eo[i0:i1, None] - ev[None, :])).ravel()
        Pi += (L * f) @ L.T
    eye = np.eye(naux)
    W = np.linalg.inv(eye - Pi)                  # W(0) in the L basis: (pq|W|rs) = Lpq^T W Lrs
    Wt = np.linalg.inv(eye - U.T @ Pi @ U)       # W~(0) in the L~ basis

    norbs = Lrows.shape[1]
    shift = np.empty(norbs)
    for k in range(norbs):
        a = np.asarray(Lrows[:, k, :])
        at = U.T @ a
        v_m = np.einsum('Pm,Pm->m', a, a)
        w_m = np.einsum('Pm,Pm->m', a, W @ a)
        wt_m = np.einsum('Pm,Pm->m', at, Wt @ at)
        sx = -w_m[:nocc].sum()
        sxt = -wt_m[:nocc].sum()
        coh = 0.5 * (w_m - v_m).sum()
        coht = 0.5 * (wt_m - v_m).sum()
        shift[k] = (sxt + coht) - (sx + coh)
    return shift


def get_cohsex_shift(gw, U, with_df=None, orbs_frz=None):
    '''Delta-COHSEX shifts of the GW orbitals (gw.orbs) from the mean-field spectrum.'''
    mf = gw._scf
    set_frozen_orbs(gw)
    if orbs_frz is None:
        orbs_frz = list(gw.orbs_frz)
    nocc = gw.nocc
    mo_coeff = _mo_without_core(gw, np.asarray(mf.mo_coeff))
    mo_energy = _mo_energy_without_core(gw, np.asarray(mf.mo_energy))
    nmo = mo_coeff.shape[1]
    if with_df is None:
        with_df = gw.with_df
    pairs = {'ov': (slice(0, nocc), slice(nocc, nmo)), 'rows': (list(orbs_frz), slice(0, nmo))}
    with temporary_env(with_df, verbose=0), temporary_env(gw.mol, verbose=0):
        blk = transform_df_pairs(with_df, mo_coeff, pairs, gw.max_memory)
    return cohsex_shift(blk['ov'], blk['rows'], mo_energy, nocc, U)


# --------------------------------------------------------------- GW drivers

def _supported_classes():
    from pyscf.gw.gw_ac_lowmem import GWACLowMem
    from pyscf.gw.gw_cd_lowmem import GWCDLowMem
    from pyscf.gw.gw_fscd import GWFSCD
    from pyscf.gw.evgw_lowmem import EVGWLowMem
    from pyscf.gw.evgw_fscd import EVGWFSCD
    return GWACLowMem, GWCDLowMem, GWFSCD, EVGWLowMem, EVGWFSCD


def _check_supported(gw):
    from pyscf.gw.qsgw_lowmem import QSGWLowMem
    if isinstance(gw, QSGWLowMem):
        raise NotImplementedError('PCM is not implemented for qsGW')
    if not isinstance(gw, _supported_classes()):
        raise NotImplementedError(
            'GW-PCM supports the pyscf-forge drivers GWACLowMem, GWCDLowMem, GWFSCD, EVGWLowMem '
            'and EVGWFSCD, got %s' % type(gw))
    if np.asarray(gw._scf.mo_energy).ndim != 1:
        raise NotImplementedError('GW-PCM is implemented for restricted references only')


class _GWPCM:
    '''Mixin for the GW drivers (see GW_PCM).'''

    _keys = {'with_solvent', 'pcm_scheme', 'eps_fast', 'pcm_U', 'pcm_shift', 'bare_df', '_pcm_vhf_df'}

    def initialize_df(self, auxbasis=None):
        if self.pcm_U is not None:       # already set up (the drivers call this in kernel)
            return
        super().initialize_df(auxbasis=auxbasis)
        self._dress()

    def _dress(self):
        '''Reaction-field metric and, for scheme 'gw', the dressed DF object.'''
        if self.pcm_U is not None:
            return
        from pyscf.gw.gw_fscd import GWFSCD
        self.bare_df = self.with_df
        with temporary_env(self.with_solvent, verbose=self.verbose, stdout=self.stdout):
            self.pcm_U, _ = get_reaction_metric(self.bare_df, self.with_solvent, self.eps_fast)
        if self.pcm_scheme == 'gw':
            self.with_df = make_dressed_df(self.bare_df, self.pcm_U, self.max_memory)
            if isinstance(self.with_df._cderi, str):
                # ao2mo needs the AO tensor in memory; use the drivers' block-wise paths
                if isinstance(self, GWFSCD):
                    self.lowmem_ints = True
                elif hasattr(self, 'outcore') and hasattr(self, 'loop_ao2mo') and \
                        not isinstance(self, _gwcd_lowmem()):
                    self.outcore = True
                else:
                    raise MemoryError('The dressed DF tensor does not fit into max_memory; '
                                      'use GWFSCD(lowmem_ints=True) or GWACLowMem')

    def _sigma_reaction(self, mo_coeff):
        '''Static self-energy of the reaction field v_reac = v~ - v (MO basis), and the bare DF
        exchange self-energy.

        The PCM response is the low-frequency limit of the electronic response of the solvent,
        whose excitation energies are far above those of the solute.  The frequency integral
        of G times such a response gives the static SEX + COH form
            Sigma^reac_pq = -sum_i (pi|v_reac|iq) + 1/2 sum_m (pm|v_reac|mq),
        with m over the (non-frozen) MO basis.  Treating v_reac as part of the bare exchange
        instead would only give the first term and leave the virtual levels unpolarized.
        '''
        dm = self._scf.make_rdm1()
        dm_all = mo_coeff @ mo_coeff.T
        dms = np.asarray([dm, dm_all])
        with temporary_env(self.bare_df, verbose=0):
            vk0 = self.bare_df.get_jk(dms, hermi=1, with_j=False)[1]
        with temporary_env(self.with_df, verbose=0):
            vk1 = self.with_df.get_jk(dms, hermi=1, with_j=False)[1]
        dk = vk1 - vk0
        sigma = mo_coeff.T @ (-0.5 * dk[0] + 0.5 * dk[1]) @ mo_coeff
        vk_df = -0.5 * mo_coeff.T @ vk0[0] @ mo_coeff
        return sigma, vk_df

    def get_sigma_exchange(self, mo_coeff):
        '''Exchange self-energy; for scheme 'gw' plus the static reaction-field self-energy
        (vhf_df is then handled here, see GW_PCM).'''
        if self.pcm_scheme != 'gw':
            return super().get_sigma_exchange(mo_coeff)
        sigma, vk_df = self._sigma_reaction(mo_coeff)
        vk = vk_df if self._pcm_vhf_df else super().get_sigma_exchange(mo_coeff)
        return vk + sigma

    def get_ham_hf(self, mo_coeff, Lpq=None):
        '''Static part of the evGW quasiparticle equation (EVGWLowMem): the mean-field Fock
        matrix (including the ground-state solvent potential) with vxc replaced by the
        exchange self-energy (with v~ for scheme 'gw'), plus the Delta-COHSEX shifts.'''
        mf = self._scf
        nocc = self.nocc
        with temporary_env(mf, verbose=0):
            dm = mf.make_rdm1()
            fock = mo_coeff.T @ mf.get_fock(dm=dm) @ mo_coeff
            vxc = mo_coeff.T @ (mf.get_veff(dm=dm) - mf.get_j(dm=dm)) @ mo_coeff
        if self.vhf_df and Lpq is not None:
            vk = -lib.einsum('Lpi,Liq->pq', Lpq[:, :, :nocc], Lpq[:, :nocc, :])
        else:
            vk = self.get_sigma_exchange(mo_coeff=mo_coeff)
        ham_hf = fock - vxc + vk
        if self.pcm_scheme == 'cohsex':
            ham_hf[self.orbs_frz, self.orbs_frz] += self._get_shift()
        return ham_hf, vk

    def _get_shift(self):
        if self.pcm_shift is None:
            self.pcm_shift = get_cohsex_shift(self, self.pcm_U, with_df=self.bare_df)
            logger.info(self, 'GW-PCM Delta-COHSEX shifts (eV) of orbitals %s:\n%s', list(self.orbs),
                        self.pcm_shift * 27.211386245988)
        return self.pcm_shift

    def kernel(self, *args, **kwargs):
        from pyscf.gw.evgw_lowmem import EVGWLowMem
        from pyscf.gw.gw_fscd import GWFSCD, prepare
        from pyscf.gw.evgw_fscd import EVGWFSCD
        if self.Lpq is not None and self.pcm_U is None:
            raise RuntimeError('GW-PCM builds its own integrals; set gw.Lpq = None')
        self.initialize_df(auxbasis=self.auxbasis)
        if isinstance(self, EVGWFSCD) and self.pcm_scheme == 'cohsex':
            # evGW + Delta-COHSEX: the static shift enters every cycle's QP equation, through
            # the exchange term (only its diagonal is used by solve_qp)
            self._check_input()
            prepare(self)
            vk = np.array(self.vk, copy=True)
            vk[self.orbs_frz, self.orbs_frz] += self._get_shift()
            self.vk = vk
        out = super().kernel(*args, **kwargs)
        if self.pcm_scheme == 'cohsex' and not isinstance(self, (EVGWLowMem, EVGWFSCD)):
            self._apply_shift()
            if isinstance(self, GWFSCD):
                out = self.mo_energy
        return out

    def _apply_shift(self):
        '''G0W0 + Delta-COHSEX (eq. 24 of the GW paper): additive shift of the QP energies.'''
        shift = self._get_shift()
        mo_energy = np.array(self.mo_energy, copy=True)
        mo_energy[list(self.orbs)] += shift
        self.mo_energy = mo_energy


def GW_PCM(gw, scheme='cohsex', eps_fast=None, solvent_obj=None):
    '''Non-equilibrium PCM solvation for a pyscf-forge GW object.

    Args:
        gw : GWACLowMem, GWCDLowMem, GWFSCD, EVGWLowMem or EVGWFSCD object, built on a
            mean-field object with PCM (the ground-state, slow solvent response)
        scheme : 'cohsex' (GW + Delta-COHSEX, default) or 'gw' (GW with the dressed v~)
        eps_fast : dielectric constant of the fast response (default: eps_optical of the
            solvent object)
        solvent_obj : PCM object (default gw._scf.with_solvent)

    Returns:
        A copy of gw whose kernel includes the fast solvent response.  Attributes:
        pcm_U (dressed metric factor), pcm_shift (Delta-COHSEX shifts of gw.orbs, a.u.),
        bare_df (density fitting with v); with scheme 'gw', with_df is dressed.
    '''
    _check_supported(gw)
    if scheme not in ('cohsex', 'gw'):
        raise ValueError('Unknown GW-PCM scheme %s' % scheme)
    if solvent_obj is None:
        solvent_obj = getattr(gw._scf, 'with_solvent', None)
        if solvent_obj is None:
            raise ValueError('GW-PCM needs a mean-field object with PCM or a solvent_obj')
    if gw.Lpq is not None:
        raise RuntimeError('GW-PCM builds its own integrals; set gw.Lpq = None')
    from pyscf.gw.gw_fscd import GWFSCD
    if isinstance(gw, GWFSCD) and gw._fs_ints is not None:
        raise RuntimeError('GW-PCM needs a GWFSCD object that has not been run yet')
    obj = lib.set_class(gw.copy(), (_GWPCM, gw.__class__))
    obj.with_solvent = solvent_obj
    obj.pcm_scheme = scheme
    obj.eps_fast = eps_fast
    obj.pcm_U = None
    obj.pcm_shift = None
    obj.bare_df = None
    obj._pcm_vhf_df = gw.vhf_df
    if scheme == 'gw':
        # the drivers bypass get_sigma_exchange with vhf_df; the static reaction self-energy
        # is added there, so the bare DF exchange (vhf_df) is evaluated there as well
        obj.vhf_df = False
    return obj


def _gwcd_lowmem():
    from pyscf.gw.gw_cd_lowmem import GWCDLowMem
    return GWCDLowMem


# --------------------------------------------------------------- BSE

def make_bse(gw, **kwargs):
    '''BSELowMem object for a GW-PCM object: quasiparticle energies from gw, screened and
    exchange kernels with the reaction-field-dressed Coulomb interaction (BSE paper, eq. 17).

    kwargs are passed to BSELowMem.  The result can also be passed to BSEQuadraticResponse.
    '''
    from pyscf.gw.bse_lowmem import BSELowMem
    if not isinstance(gw, _GWPCM) or gw.pcm_U is None:
        raise ValueError('make_bse needs a GW-PCM object (GW_PCM) that has been run')
    if gw.pcm_scheme == 'gw':
        # gw.Lpq (if any) and gw.with_df already carry v~
        return BSELowMem(gw, **kwargs)
    if gw.Lpq is not None:
        with temporary_env(gw, Lpq=dress_lpq(gw.Lpq, gw.pcm_U)):
            return BSELowMem(gw, **kwargs)
    dressed = make_dressed_df(gw.bare_df, gw.pcm_U, gw.max_memory)
    with temporary_env(gw, with_df=dressed):
        return BSELowMem(gw, **kwargs)
