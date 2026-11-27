#!/usr/bin/env python

'''
GW and BSE in solution: non-equilibrium polarizable continuum model.

The ground-state (slow) solvent response comes from the PCM mean-field calculation with the
static dielectric constant.  The fast (electronic) response, with the optical dielectric
constant, screens the quasiparticle and optical excitations:

    scheme='cohsex' : GW + Delta-COHSEX (default; Duchemin et al., JCP 144, 164106 (2016))
    scheme='gw'     : GW with the reaction-field-dressed Coulomb interaction

BSE (Duchemin et al., Chem. Sci. 9, 4430 (2018)) uses the GW-PCM quasiparticle energies and
the dressed screened and exchange kernels; the BSE object also drives BSE quadratic response.
'''

import numpy as np
from pyscf import gto, dft
from pyscf.gw.gw_fscd import GWFSCD
from pyscf.gw.qp_shift import scissor_shift
from pyscf.gw.bse_qr import BSEQuadraticResponse
from pyscf.gw import pcm_gw

HA2EV = 27.211386245988

mol = gto.M(atom='''
C   0.000000   0.000000  -0.602900
O   0.000000   0.000000   0.601500
H   0.000000   0.936400  -1.184000
H   0.000000  -0.936400  -1.184000''', basis='def2-svp', verbose=4)

# ground state in water: eps = 78.3553 (static); eps_optical = 1.78 is the fast response
mf = dft.RKS(mol, xc='pbe0').PCM()
mf.with_solvent.method = 'IEF-PCM'
mf.with_solvent.solvent = 'water'
mf.kernel()

nocc = mol.nelectron // 2
orbs = list(range(nocc - 3, nocc + 3))


def run_gw(scheme, eps_fast=None):
    gw = GWFSCD(mf, auxbasis='def2-svp-ri')
    gw.orbs = orbs
    gw = pcm_gw.GW_PCM(gw, scheme=scheme, eps_fast=eps_fast)
    gw.kernel()
    scissor_shift(gw, inplace=True)   # orbitals outside gw.orbs, for BSE
    return gw


# eps_fast = 1 switches off the fast response (ground-state solvation only)
gw_slow = run_gw('cohsex', eps_fast=1.0)
gw_coh = run_gw('cohsex')
gw_gw = run_gw('gw')

homo, lumo = nocc - 1, nocc
for label, gw in (('ground-state PCM only', gw_slow), ('GW + Delta-COHSEX', gw_coh), ('Delta-GW', gw_gw)):
    e = np.asarray(gw.mo_energy) * HA2EV
    print('%-24s IP = %7.3f eV  EA = %7.3f eV  gap = %7.3f eV' % (label, -e[homo], -e[lumo], e[lumo] - e[homo]))
print('Delta-COHSEX shifts of orbitals %s (eV): %s' % (orbs, np.round(gw_coh.pcm_shift * HA2EV, 3)))

# BSE with the dressed kernels
for label, gw in (('ground-state PCM only', gw_slow), ('GW + Delta-COHSEX', gw_coh)):
    bse = pcm_gw.make_bse(gw)
    bse.nroot = 4
    bse.kernel('s')
    print('%-24s BSE singlets (eV): %s' % (label, np.round(np.asarray(bse.exci).ravel()[:4] * HA2EV, 3)))

# static polarizability with BSE linear response in solution (experimental)
alpha = BSEQuadraticResponse(pcm_gw.make_bse(gw_coh)).polarizability()
print('BSE-PCM static polarizability (a.u.):\n', alpha)
