#!/usr/bin/env python

'''
First hyperpolarizability and two-photon absorption from the static-screened
Bethe-Salpeter equation (BSE@G0W0), neglecting the BSE hyperkernel.

Ref: N. Rauwolf, W. Klopper, C. Holzer, J. Chem. Phys. 160, 061101 (2024)
'''

from pyscf import gto, dft
from pyscf.gw.gw_ac import GWAC
from pyscf.gw.bse_qr import QRBSE, beta_parallel

mol = gto.M(
    atom='O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692',
    basis='aug-cc-pvdz', verbose=4)
mf = dft.RKS(mol, xc='pbe0').run()

gw = GWAC(mf)
# By default all orbitals are corrected.  For larger systems restrict GW to a window around
# the gap; QRBSE then fills in the other orbitals with a scissor shift
# (pyscf.gw.qp_shift.scissor_shift), e.g.
#   nocc = mol.nelectron // 2
#   gw.orbs = range(nocc - 4, nocc + 8)
gw.kernel()

qr = QRBSE(gw)

# static polarizability and first hyperpolarizability
alpha = qr.polarizability()
beta0 = qr.hyperpolarizability()

# second-harmonic generation at 1064 nm, beta(-2w; w, w)
w = 0.0428
beta_shg = qr.hyperpolarizability(w, w)

mu = mf.dip_moment()
print('beta_|| static       = %.3f a.u.' % beta_parallel(beta0, mu))
print('beta_|| SHG 1064 nm  = %.3f a.u.' % beta_parallel(beta_shg, mu))

# degenerate two-photon absorption for the lowest singlet states
exci, tensors, delta = qr.two_photon_absorption(nstates=5)
