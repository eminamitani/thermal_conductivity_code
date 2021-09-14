from ase.io import read,write
vasp=read('optimized.vasp',format='vasp')
write('optimized.data',vasp,format='lammps-data')