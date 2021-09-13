from ase.io import read,write
atoms=read('optimized.vasp',format='vasp')
write('data.si',atoms,format='lammps-data')

'''
from lammps import lammps
lmp = lammps()
lmp.file('in.lammps')
lmp.command("min_style sd")
lmp.command("minimize 1.0e-4 1.0e-6 200 1000")
lmp.command("write_data optimized.data")
'''
from phonolammps import Phonolammps

phlammps = Phonolammps('in.phonon',
                       supercell_matrix=[[1, 0, 0],
                                         [0, 1, 0],
                                         [0, 0, 1]],
                       primitive_matrix=[[1.0, 0.0 ,0.0],
                                         [0.0, 1.0, 0.0],
                                         [0.0, 0.0, 1.0]])

unitcell = phlammps.get_unitcell()
force_constants = phlammps.get_force_constants()
supercell_matrix = phlammps.get_supercell_matrix()

#writing down force_constant
phlammps.write_force_constants('force_constants.txt')
phlammps.write_force_constants('force_constants.hdf5',hdf5=True)
phlammps.write_unitcell_POSCAR()