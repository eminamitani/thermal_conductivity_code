#data parse part 

#reading force constant data from phonopy format
#scaled by mass, thus retun is dynamical matrix
import numpy as np
def read_fc_phonopy(FC_file,natom,masses):
    """
    Reading force constant data from phonopy format.
    In this module, force constant is scaled by mass.
    Thus, the return is dynamical matrix
    """

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

def read_fc_phonopy_noscale(FC_file,natom):
    """
    Reading force constant data from phonopy format.
    In this module, force constant is not scaled by mass.
    Thus, the return is phonopy format force constant ndarray
    """
    with open(FC_file,'r') as fc:
        lines=fc.readlines()

        nlineblock=4
        fc_all=np.zeros((natom,natom,3,3))
        start=1
        for i in range(natom):
            for j in range(natom):
                fc_block=lines[start+1:start+nlineblock]
                fc=np.loadtxt(fc_block)
                fc_all[i,j]=fc
                start=start+nlineblock
    
    return fc_all


#convert phonopy style to flat(LAMMPS Dyn regular) format
def phonopy_to_flat(force_constants, natom):
    return np.reshape(force_constants.transpose(0,2,1,3),(natom*3,natom*3))

#convert flat to phonopy style
def flat_to_phonopy(force_constants,natom):
    return np.reshape(force_constants,(natom,3,natom,3)).transpose(0,2,1,3)


def dynmat_to_fcphonopy(dynmat,natom,masses):
    '''
    convert from dynamical matrix in ndarray(natom,natom,3,3) to force constant
    return: phonopy format force constant
    ''' 
    fcphonopy=np.zeros((natom,natom,3,3))
    for i in range(natom):
        for j in range(natom):
            fcphonopy[i,j]=dynmat[i,j]*np.sqrt(masses[i]*masses[j])
    
    return fcphonopy

def fcphonopy_to_dynmat(force_constants,natom,masses):
    dynmat=np.zeros((natom,natom,3,3))
    for i in range(natom):
        for j in range(natom):
            dynmat[i,j]=force_constants[i,j]/np.sqrt(masses[i]*masses[j])
    
    return dynmat

#for symmetrization to supress negative mode
#using phonopy API
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms

def symmetrize_phonopy(atoms,FC_file):
    positions=atoms.positions
    masses=atoms.get_masses()
    natom=len(positions)
    unitcell = PhonopyAtoms(symbols=atoms.symbols,
                        cell=atoms.cell,
                        positions=atoms.positions)
    phonon = Phonopy(unitcell,
                 [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                 primitive_matrix=[[1,0,0],[0,1,0],[0,0,1]],log_level=1)
    
    #read as ndarray, no mass scaling here
    fc_phonopy=read_fc_phonopy_noscale(FC_file,natom)
    
    phonon.set_force_constants(fc_phonopy)
    phonon.symmetrize_force_constants(show_drift=True)
    symetrized_fc=phonon.force_constants
    #primaly check of frequency
    mesh = [1, 1, 1]
    phonon.run_mesh(mesh)
    mesh_dict = phonon.get_mesh_dict()
    frequencies = mesh_dict['frequencies']
    np.savetxt('frequency_symmetrized_fc.txt',frequencies)

    #scale by mass
    symmetrized_dyn=fcphonopy_to_dynmat(symetrized_fc,natom,masses)
    
    #return flat form dynamical matrix
    return phonopy_to_flat(symmetrized_dyn,natom=natom)

def symmetrize_lammps(atoms,FC_file):
    positions=atoms.positions
    masses=atoms.get_masses()
    natom=len(positions)
    unitcell = PhonopyAtoms(symbols=atoms.symbols,
                        cell=atoms.cell,
                        positions=atoms.positions)
    phonon = Phonopy(unitcell,
                 [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                 primitive_matrix=[[1,0,0],[0,1,0],[0,0,1]],log_level=1)
    
    #read as ndarray
    nmodes=natom*3
    lammps_dyn=np.loadtxt(FC_file).reshape((nmodes,nmodes))
    converted_dyn=flat_to_phonopy(lammps_dyn,natom)
    
    #convert mass scaled dynamical matrix to force constant form
    fc_phonopy=dynmat_to_fcphonopy(converted_dyn,natom,masses)
    
    phonon.set_force_constants(fc_phonopy)
    phonon.symmetrize_force_constants(show_drift=True)
    symetrized_fc=phonon.force_constants
    #primaly check of frequency
    mesh = [1, 1, 1]
    phonon.run_mesh(mesh)
    mesh_dict = phonon.get_mesh_dict()
    frequencies = mesh_dict['frequencies']
    np.savetxt('frequency_symmetrized_fc.txt',frequencies)

    #scale by mass
    symmetrized_dyn=fcphonopy_to_dynmat(symetrized_fc,natom,masses)
    
    #return flat form dynamical matrix
    return phonopy_to_flat(symmetrized_dyn,natom=natom)




