#!/usr/bin/env python

'''
Eigenvalue self-consistent GW (evGW) with frequency-sampled contour deformation, for a core
level plus a valence window.

Orbitals outside gw.orbs are scissor-shifted every cycle (each takes the correction of the
energetically closest computed orbital of the same kind) and enter both G and the screening
with their shifted energies.

Ref: M. Kehry, W. Klopper, C. Holzer, J. Chem. Phys. 159, 044116 (2023)
'''

from pyscf import gto, dft
from pyscf.gw.evgw_fscd import EVGWFSCD

mol = gto.M(atom='C 0 0 0; O 0 0 1.128', basis='def2-tzvp', verbose=4)
mf = dft.RKS(mol, xc='pbe0').density_fit().run()

nocc = mol.nelectron // 2
gw = EVGWFSCD(mf)
gw.orbs = [0, 1] + list(range(nocc - 3, nocc + 3))   # O 1s, C 1s and a valence window
gw.fs_exact_cutoff = 2.0     # exact small-frequency residues: agrees with canonical CD
# gw.W0 = True               # evGW0: screening kept at the DFT energies
gw.kernel()

H = 27.211386
for p, label in ((0, 'O 1s'), (1, 'C 1s'), (nocc - 1, 'HOMO'), (nocc, 'LUMO')):
    print('%-5s  KS %10.3f eV   evGW %10.3f eV' % (label, mf.mo_energy[p] * H, gw.mo_energy[p] * H))

# gw.mo_energy holds the complete spectrum (window from GW, the rest scissor-shifted) and can
# be passed on directly, e.g. to pyscf.gw.bse_qr
