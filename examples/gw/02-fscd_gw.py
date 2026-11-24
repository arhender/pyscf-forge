#!/usr/bin/env python

'''
G0W0 core ionization energies with frequency-sampled contour deformation (fsCD-GW).

Canonical contour deformation (pyscf.gw.gw_cd.GWCD) needs one dielectric matrix per residue,
so the cost of a core state grows with the number of occupied orbitals above it.  fsCD-GW
evaluates W only on the imaginary-axis quadrature grid plus a sparse, adaptive set of real
frequencies and obtains the residues from AAA rational fits.

Ref: M. Kehry, W. Klopper, C. Holzer, J. Chem. Phys. 159, 044116 (2023)
'''

from pyscf import gto, dft
from pyscf.gw.gw_fscd import GWFSCD

mol = gto.M(atom='C 0 0 0; O 0 0 1.128', basis='def2-tzvp', verbose=4)
mf = dft.RKS(mol, xc='pbe0').density_fit().run()

nocc = mol.nelectron // 2
gw = GWFSCD(mf)
gw.orbs = [0, 1, nocc - 1, nocc]        # O 1s, C 1s, HOMO, LUMO
# Pure frequency sampling (the paper's scheme) is the default.  Evaluating the few residues
# at small frequencies exactly makes the result agree with canonical CD to ~1 meV at
# essentially the same cost for core states:
gw.fs_exact_cutoff = 2.0
gw.kernel()

for p, label in zip(gw.orbs, ('O 1s', 'C 1s', 'HOMO', 'LUMO')):
    print('%-5s  KS %10.3f eV   G0W0 %10.3f eV' % (label, mf.mo_energy[p] * 27.211386, gw.mo_energy[p] * 27.211386))

# Sigma^c(omega) can be inspected directly, e.g. around the O 1s quasiparticle peak
import numpy as np
omegas = gw.mo_energy[0] + np.linspace(-0.1, 0.1, 5)
print(gw.get_sigma_c(0, omegas).real * 27.211386)
