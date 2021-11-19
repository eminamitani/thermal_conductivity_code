from ase.io import read
from tqdm import tqdm

atoms=read('/home/emi/lammps/aSi-4096/optimized.data',format='lammps-data',style='atomic')
import numpy as np
natom=len(atoms.positions)
nmodes=natom*3
print('loading data...')
dyn_file='/home/emi/lammps/aSi-4096/Dyn.form'
lammps_dyn=np.loadtxt(dyn_file).reshape((nmodes,nmodes))
eigenvalue, eigenvector=np.linalg.eigh(lammps_dyn)
print('finish eigenvector/eigenvalue calculation')

kmin=2.0*np.pi/atoms.cell[0,0]
kmax=kmin*10
kvec=np.linspace(kmin,kmax,10)
n_kpt=len(kvec)
import pyAF.postprocess
n_sample=10
n_modes=natom*3
C_L=np.zeros((n_kpt,n_modes))
C_T=np.zeros((n_kpt,n_modes))
position=atoms.positions

for ik,k in enumerate(tqdm(kvec)):
    k_vector=pyAF.postprocess.polar_coord_sampling(n_sample,k)

    for imode in range(nmodes):
        ev=eigenvector[:,imode]
        cl, ct=pyAF.postprocess.get_Ci_ver3(ev,position,k)
        C_L[ik,imode]=cl
        C_T[ik,imode]=ct

np.save('C_L_4096_v3',C_L)
np.save('C_T_4096_v3',C_T)
