#data parse part 

#reading force constant data from phonopy format
import numpy as np
def read_fc_phonopy(FC_file,natom,masses):

    with open(FC_file,'r') as fc:
        lines=fc.readlines()

        nlineblock=4
        fc_all=np.zeros((natom,natom,3,3))
        start=1
        for i in range(natom):
            for j in range(natom):
                fc_block=lines[start+1:start+nlineblock]
                fc=np.loadtxt(fc_block)
                fc_all[i,j]=fc/np.sqrt(masses[i]*masses[j])
                #fc_all[i,j]=fc
                start=start+nlineblock
    
    return fc_all

#convert phonopy style to flat(LAMMPS Dyn regular) format
def phonopy_to_flat(force_constants, natom):
    return np.reshape(force_constants.transpose(0,2,1,3),(natom*3,natom*3))

#convert flat to phonopy style
def flat_to_phonopy(force_constants,natom):
    return np.reshape(force_constants,(natom,3,natom,3)).transpose(0,2,1,3)
