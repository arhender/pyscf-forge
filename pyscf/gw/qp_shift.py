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
Scissor shift for orbitals outside the GW quasiparticle window.

pyscf.gw.gw_ac.GWAC only solves the quasiparticle equation for the orbitals in gw.orbs and
leaves all other entries of gw.mo_energy at zero.  Before using such energies in a BSE (or
response) calculation, the remaining orbitals have to be filled in.  The common practice
(e.g. Turbomole's "gap" option, which computes only HOMO and LUMO and shifts the rest) is a
rigid shift applied separately to occupied and virtual orbitals:

    eps_p^QP = eps_p^MF + Delta_q

where q is the computed orbital of the same kind (occupied / virtual) closest in mean-field
energy to p.  For the usual contiguous window around the gap this means every occupied
orbital below the window receives the correction of the lowest computed occupied orbital and
every virtual above the window receives the correction of the highest computed virtual
orbital.  For non-contiguous windows (e.g. a core level plus a valence window) each orbital
takes the correction of the energetically closest computed orbital, so valence orbitals are
not shifted by a core correction.
'''

import numpy as np
from pyscf.lib import logger


def scissor_shift(gw, orbs=None, mo_energy_qp=None, mo_energy_mf=None, mo_occ=None, inplace=False):
    '''Fill in quasiparticle energies of orbitals not treated explicitly by GW.

    Args:
        gw : restricted GW object (e.g. GWAC) after kernel().

    Kwargs:
        orbs : list of int
            Indices (in the full MO space) of explicitly corrected orbitals. Default gw.orbs.
        mo_energy_qp, mo_energy_mf, mo_occ : 1D arrays
            Override gw.mo_energy, gw._scf.mo_energy and gw._scf.mo_occ.
        inplace : bool
            If True, also store the result in gw.mo_energy.

    Returns:
        mo_energy : 1D array of quasiparticle energies for all orbitals.
    '''
    if mo_energy_qp is None:
        mo_energy_qp = gw.mo_energy
    if mo_energy_mf is None:
        mo_energy_mf = gw._scf.mo_energy
    if mo_occ is None:
        mo_occ = gw._scf.mo_occ
    if orbs is None:
        orbs = gw.orbs
    mo_energy_qp = np.asarray(mo_energy_qp, dtype=float)
    mo_energy_mf = np.asarray(mo_energy_mf, dtype=float)
    mo_occ = np.asarray(mo_occ)
    if mo_energy_qp.ndim != 1 or mo_energy_mf.ndim != 1:
        raise NotImplementedError('scissor_shift supports restricted references only.')
    if getattr(gw, 'frozen', None) is not None:
        raise NotImplementedError('scissor_shift does not support frozen orbitals.')

    mo_energy = scissor_shift_energies(mo_energy_qp, mo_energy_mf, mo_occ, orbs,
                                       log=logger.new_logger(gw))
    if inplace:
        gw.mo_energy = mo_energy
    return mo_energy


def scissor_shift_energies(mo_energy_qp, mo_energy_mf, mo_occ, orbs=None, log=None):
    '''Array-level scissor shift (see scissor_shift).

    Orbitals in `orbs` take their values from mo_energy_qp; every other occupied (virtual)
    orbital is shifted by the correction mo_energy_qp - mo_energy_mf of the computed occupied
    (virtual) orbital closest in mean-field energy.  All arrays refer to the same orbital space, which may
    be a frozen-core subspace.
    '''
    mo_energy_qp = np.asarray(mo_energy_qp, dtype=float)
    mo_energy_mf = np.asarray(mo_energy_mf, dtype=float)
    nmo = mo_energy_mf.size
    if orbs is None:
        orbs = range(nmo)
    orbs = np.unique(np.asarray(list(orbs), dtype=int))
    occ = np.asarray(mo_occ) > 0
    computed = np.zeros(nmo, dtype=bool)
    computed[orbs] = True

    mo_energy = mo_energy_mf.copy()
    mo_energy[computed] = mo_energy_qp[computed]
    corr = mo_energy_qp - mo_energy_mf
    for kind, mask in (('occupied', occ), ('virtual', ~occ)):
        ref = np.where(computed & mask)[0]
        todo = np.where(~computed & mask)[0]
        if todo.size == 0:
            continue
        if ref.size == 0:
            raise RuntimeError('No %s orbital was corrected by GW; cannot shift the %s orbitals. '
                               'Include at least one %s orbital in gw.orbs.' % (kind, kind, kind))
        nearest = ref[np.abs(mo_energy_mf[todo][:, None] - mo_energy_mf[ref][None, :]).argmin(axis=1)]
        mo_energy[todo] = mo_energy_mf[todo] + corr[nearest]
        if log is not None:
            for q in np.unique(nearest):
                log.info('scissor shift: %d %s orbital(s) shifted by %.6f Ha (correction of orbital %d)',
                         np.count_nonzero(nearest == q), kind, corr[q], q)
    return mo_energy
