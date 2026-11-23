#!/usr/bin/env python

'''
First hyperpolarizability and two-photon absorption from TDDFT quadratic response
(adiabatic, including the exchange-correlation hyperkernel).

Ref: S. M. Parker, D. Rappoport, F. Furche, J. Chem. Theory Comput. 14, 807 (2018)
'''

from pyscf import gto, dft
from pyscf.tdscf.quadratic_response import QRTDDFT, beta_parallel

mol = gto.M(
    atom='O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692',
    basis='aug-cc-pvdz', verbose=4)
mf = dft.RKS(mol, xc='pbe0').run()

qr = QRTDDFT(mf)

# static first hyperpolarizability
beta0 = qr.hyperpolarizability()

# second-harmonic generation at 1064 nm, beta(-2w; w, w)
w = 0.0428
beta_shg = qr.hyperpolarizability(w, w)

mu = mf.dip_moment()
print('beta_|| static       = %.3f a.u.' % beta_parallel(beta0, mu))
print('beta_|| SHG 1064 nm  = %.3f a.u.' % beta_parallel(beta_shg, mu))

# degenerate two-photon absorption for the lowest singlet states
exci, tensors, delta = qr.two_photon_absorption(nstates=5)
