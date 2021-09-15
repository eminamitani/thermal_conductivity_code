# pyAF: thermal conductivity calculation based on Allen-Feldman theory
This python package include modules to evaluate thermal conductivity in disordered system from Allen-Feldman theory[1].
The dynamical matrix file from LAMMPS and force constant file of phonopy format can be used as input.

## Requirements
numpy, ASE, pyyaml

## Install
At the directory where setup.py exist, type the following command
```
pip install -e .
```

## Usage
First, dynamical matrix or force constant information is required.
### example of LAMMPS input
This is the input of LAMMPS to form Dyn.form (dynamical matrix) in aSi512-test directory.
```
units           metal
boundary        p p p
atom_style      atomic
atom_modify     map array sort 0 0.0
read_data  optimized.data
mass   1        28.0855000000 
pair_style	 sw
pair_coeff 	 * * Si.sw Si
neighbor    3.0 bin
dynamical_matrix all regular 1.0e-6 file Dyn.form binary no 
```

### evaluate thermal conductivity via interface
pyAF use yaml file to get the computational setup. For example, `setup.yaml` has following lines.
```
structure_file: 'optimized.vasp' #file name of VASP POSCAR format of unitcell information
dyn_file: 'Dyn.form'  #file name of dynamical matrix
style: 'lammps-regular' #lammps-regular or phonopy
temperature: 300        #temperature to evaluate thermal conductivity
broadening_factor: 5.0  #Lorentzian width
using_mean_spacing: True #use mean spacing of frequency in smearing. If True, broadening_factor*average_spacing is used as the width of Lorentzian
omega_threshould: 0.01 #minimum frequency to take into account
broadening_threshould: 0.01 #minimum value of Lorentzian weight to take into account
two_dim: False #two dimensional or three dimensional
```

You can get averaged thermal conductivity by the following script.
```
from pyAF.interface import thermal_conductivity
results=thermal_conductivity('setup.yaml')
import numpy as np
kappa=np.sum(results['thermal_conductivity'])
print(kappa)
```

`interface.thermal_conductivity` returns the dictionary of `{'freq','diffusivity','thermal_conductivity'}` per mode.
`interface.resolved_thermal_conductivity` returns the dictionary of `{'freq','diffusivity','thermal_conductivity'}` per mode. 
But, diffusivity and thermal_conductivity is not averaged, thus they have three elements per mode (x,y,z components).


## Ref
[1]Philip B. Allen and Joseph L. Feldman. Thermal conductivity of disordered harmonic solids. Phys. Rev. B, Vol. 48, pp. 12581–12588, Nov 1993.