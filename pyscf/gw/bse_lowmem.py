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
Low-memory variant of the Bethe-Salpeter equation solver pyscf.gw.bse.BSE.

pyscf.gw.bse.BSE keeps the full MO three-center integrals Lpq (naux, nmo, nmo) (transforming
them if the GW object has none), and bse_davidson then makes contiguous copies of the
occupied-virtual and virtual-virtual blocks next to it, so the peak is about
2 * naux * nmo^2 doubles plus the screened blocks.  The Davidson algorithm only needs the
occupied-occupied, occupied-virtual and virtual-virtual blocks (the virtual-occupied block
is the transpose of the occupied-virtual one).  BSELowMem

  * transforms only these three blocks, directly from the density-fitting integrals in
    blocks of auxiliary functions (never forming the full Lpq), or copies them from the GW
    object's Lpq if it has one (free_gw_lpq=True then drops the GW object's copy);
  * exposes them as self.Lpq through a light-weight (nspin, naux, nmo, nmo) array-like
    object.  Slices that coincide with a stored block return that array itself, so the
    np.ascontiguousarray calls in pyscf.gw.bse.bse_davidson make no copies; other slices
    (e.g. the trial-vector subspace) are assembled on demand.

The Davidson, Lanczos and analysis code of pyscf.gw.bse is used unchanged and gives the
same results.  Upstream, the equivalent change would be for bse_davidson / bse_lanczos to
take the three blocks instead of Lpq.  full_diagonalization (small systems only) assembles
the full Lpq temporarily.
'''

import numpy as np

from pyscf import lib
from pyscf.lib import logger
from pyscf.ao2mo._ao2mo import nr_e2
from pyscf.gw.bse import BSE


class _SpinLpq:
    '''(naux, nmo, nmo) view of one spin's oo/ov/vv blocks; supports basic slicing.'''

    ndim = 3
    dtype = np.dtype(np.float64)

    def __init__(self, blocks, nocc, nmo):
        self.b = blocks
        self.o = nocc
        self.nmo = nmo
        self.naux = blocks['ov'].shape[0]
        self.shape = (self.naux, nmo, nmo)

    def __array__(self, dtype=None, copy=None):
        out = self[:, :, :]
        return out if dtype is None else out.astype(dtype)

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        key = key + (slice(None),) * (3 - len(key))
        if any(not isinstance(k, slice) for k in key):
            return np.asarray(self)[key]
        ka, kp, kq = key
        n, o = self.nmo, self.o
        p0, p1, sp = kp.indices(n)
        q0, q1, sq = kq.indices(n)
        if sp != 1 or sq != 1:
            return np.asarray(self)[key]
        full_aux = ka == slice(None)
        if full_aux:
            if (p0, p1, q0, q1) == (0, o, o, n):
                return self.b['ov']
            if (p0, p1, q0, q1) == (0, o, 0, o):
                return self.b['oo']
            if (p0, p1, q0, q1) == (o, n, o, n):
                return self.b['vv']
        blk = {k: v[ka] for k, v in self.b.items()}
        out = np.empty((blk['ov'].shape[0], max(p1 - p0, 0), max(q1 - q0, 0)))
        ro = (p0, min(p1, o))          # occupied rows
        rv = (max(p0, o), p1)          # virtual rows
        co = (q0, min(q1, o))
        cv = (max(q0, o), q1)

        def put(r, c, src):
            if r[1] > r[0] and c[1] > c[0]:
                out[:, r[0] - p0:r[1] - p0, c[0] - q0:c[1] - q0] = src
        put(ro, co, blk['oo'][:, ro[0]:ro[1], co[0]:co[1]])
        put(ro, cv, blk['ov'][:, ro[0]:ro[1], cv[0] - o:cv[1] - o])
        put(rv, co, blk['ov'][:, co[0]:co[1], rv[0] - o:rv[1] - o].transpose(0, 2, 1))
        put(rv, cv, blk['vv'][:, rv[0] - o:rv[1] - o, cv[0] - o:cv[1] - o])
        return out


class _BlockLpq:
    '''(nspin, naux, nmo, nmo) array-like over the per-spin blocks.'''

    ndim = 4
    dtype = np.dtype(np.float64)

    def __init__(self, spins):
        self.spins = spins
        self.shape = (len(spins),) + spins[0].shape

    @property
    def nbytes(self):
        return sum(v.nbytes for s in self.spins for v in s.b.values())

    def __len__(self):
        return len(self.spins)

    def __array__(self, dtype=None, copy=None):
        out = np.asarray([np.asarray(s) for s in self.spins])
        return out if dtype is None else out.astype(dtype)

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            return self.spins[key]
        if isinstance(key, tuple) and isinstance(key[0], (int, np.integer)):
            rest = tuple(k for k in key[1:] if k is not Ellipsis)
            return self.spins[key[0]][rest]
        return np.asarray(self)[key]


def _transform_blocks(with_df, mo_coeff, nocc, max_memory):
    '''oo, ov and vv blocks of the MO three-center integrals, transformed from the DF tensor
    in blocks of auxiliary functions (the full (naux, nmo, nmo) tensor is never formed).'''
    nao, nmo = mo_coeff.shape
    naux = with_df.get_naoaux()
    mo = np.asarray(mo_coeff, order='F')
    slices = {'oo': (0, nocc, 0, nocc), 'ov': (0, nocc, nocc, nmo), 'vv': (nocc, nmo, nocc, nmo)}
    out = {k: np.empty((naux, s[1] - s[0], s[3] - s[2])) for k, s in slices.items()}
    avail = max(max_memory - lib.current_memory()[0], 100) * 0.5e6 / 8
    blksize = int(max(1, min(naux, avail // max(nao * (nao + 1) // 2 + nmo * nmo, 1))))
    p0 = 0
    for eri1 in with_df.loop(blksize=blksize):
        p1 = p0 + eri1.shape[0]
        for k, ij in slices.items():
            nr_e2(eri1, mo, ij, aosym='s2', mosym='s1', out=out[k][p0:p1])
        p0 = p1
    return out


class BSELowMem(BSE):
    '''BSE (pyscf.gw.bse.BSE) storing only the oo/ov/vv blocks of the DF integrals.

    Same attributes, methods and results as pyscf.gw.bse.BSE; self.Lpq is an array-like
    over the stored blocks.

    Args:
        gw : restricted or unrestricted GW object (as for BSE).
        free_gw_lpq : bool
            If the GW object holds the full Lpq, set gw.Lpq = None after the blocks have been
            copied (otherwise the full tensor stays alive in the GW object).  Default False.
    '''

    def __init__(self, gw, free_gw_lpq=False):
        self.verbose = gw.verbose
        self.stdout = gw.stdout
        self.nspin = 1 if np.asarray(gw.mo_energy).ndim == 1 else 2
        self.mol = gw.mol
        self.mf = gw._scf
        self.max_memory = getattr(gw, 'max_memory', gw.mol.max_memory)
        self.nocc = np.asarray(gw.nocc)
        if self.nocc.ndim == 0:
            self.nocc = self.nocc[np.newaxis, ...]
        self.mo_energy = np.asarray(gw.mo_energy)
        if self.mo_energy.ndim == 1:
            self.mo_energy = self.mo_energy[np.newaxis, ...]
        self.mo_coeff = gw.mo_coeff
        if self.mo_coeff.ndim == 2:
            self.mo_coeff = self.mo_coeff[np.newaxis, ...]
        self.nmo = self.mo_energy.shape[-1]
        if self.nspin == 2 and isinstance(gw.nmo, int):
            gw.nmo = [gw.nmo, gw.nmo]

        Lfull = getattr(gw, 'Lpq', None)
        spins = []
        for s in range(self.nspin):
            o, n = int(self.nocc[s]), self.nmo
            if Lfull is not None:
                L = Lfull[s] if np.ndim(Lfull) == 4 else Lfull
                blocks = {'oo': np.ascontiguousarray(L[:, :o, :o]),
                          'ov': np.ascontiguousarray(L[:, :o, o:]),
                          'vv': np.ascontiguousarray(L[:, o:, o:])}
            else:
                if getattr(gw, 'with_df', None) is None:
                    gw.initialize_df(auxbasis=getattr(gw, 'auxbasis', None))
                blocks = _transform_blocks(gw.with_df, self.mo_coeff[s], o, self.max_memory)
            spins.append(_SpinLpq(blocks, o, n))
        if Lfull is not None and free_gw_lpq:
            gw.Lpq = None
        self.Lpq = _BlockLpq(spins)

        # options (as pyscf.gw.bse.BSE)
        self.TDA = False
        self.delete_lpq = False
        self.chkfile = None
        self.chk_every = 10
        self.multi = None
        self.nroot = 10
        self.trial = 'identity'
        self.nocc_sub = 50
        self.nvir_sub = 150
        self.max_vec = 12 * self.nroot
        self.max_iter = 100
        self.max_expand = min(100, self.nroot)
        self.residue_thresh = 1e-8
        self.init_ntri = min(100, self.nroot)
        self.restart_max_size = None

        self.exci = None
        self.X_vec = None
        self.Y_vec = None

    def check_memory(self):
        '''Estimate the memory of the Davidson solver: stored blocks, their screened
        oo/ov copies, and the trial-vector arrays.'''
        nvir = [(self.nmo - self.nocc[i]) for i in range(self.nspin)]
        dim = [(self.nocc[i] * nvir[i]) for i in range(self.nspin)]
        full_dim = sum(dim)
        naux = self.Lpq.shape[1]
        bar = sum(naux * (self.nocc[i]**2 + dim[i]) for i in range(self.nspin))
        mem = self.Lpq.nbytes + (bar + self.max_vec * full_dim * 3) * 8
        logger.info(self, 'BSE (lowmem) needs at least %.1f GB memory.', mem / 1.0e9)

    def full_diagonalization(self, multi):
        '''Full diagonalization (small systems): assembles the full Lpq temporarily.'''
        Lblk = self.Lpq
        self.Lpq = np.asarray(Lblk)
        try:
            return BSE.full_diagonalization(self, multi)
        finally:
            self.Lpq = Lblk

