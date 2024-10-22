# pyAF

>thermal conductivity calculation based on Allen-Feldman theory
---

This python package include modules to evaluate thermal conductivity in disordered system from Allen-Feldman theory[1].
The dynamical matrix file from LAMMPS and force constant file of phonopy format can be used as input.

## Requirements
- numpy 
- ASE 
- pyyaml
- phonopy


## Install
It is recommended to use virtual environment to avoid conflict of package name. 
The sample of preparation of virtual environment & activation is as follows.
```
python3 -m venv env
source ./env/bin/activate
```
After activate the virtual environment, clone the repository and install required package.
Since this package is not archved in PyPI and still under construction, it is recommended to install in editable mode.
```
pip3 install numpy ase pyyaml 
pip3 install phonopy
git clone https://github.com/eminamitani/thermal_conductivity_code.git 
cd ./thermal_conductivity_code 
pip3 install -e .
```

## Usage

1. obtain dynamical matrix or force constant file
2. run python script using pyAF interface

The example of amorphous Si system is stored in `aSi512_test`. The results are evaluated by comparing the calculation results from GULP(http://gulp.curtin.edu.au/gulp/) code. 

![](/sample.png)


### example of LAMMPS input to get dynamical matrix
---
This is the input of LAMMPS to form `Dyn.form` (dynamical matrix) in aSi512-test directory.
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

### input for the pyAF interface
---
pyAF use yaml file to get the computational setup. For example, `setup.yaml` has following lines.
```
structure_file: 'optimized.vasp'    #file name of VASP POSCAR format of unitcell information
dyn_file: 'Dyn.form'                #file name of dynamical matrix
style: 'lammps-regular'             #lammps-regular or phonopy
temperature: 300                    #temperature to evaluate thermal conductivity
broadening_factor: 5.0              #Lorentzian width
using_mean_spacing: True            #use mean spacing of frequency in smearing. 
                                    #If True, broadening_factor*average_spacing is used as the width of Lorentzian
omega_threshould: 0.01              #minimum frequency to take into account
broadening_threshould: 0.01         #minimum value of Lorentzian weight to take into account
two_dim: False                      #two dimensional or three dimensional
symmetrize_fc: True                #symmetrization of force constant (acoustic sum rule)
fix_diag: True                      #fix diagonal elements of Sij for 0.0. Basically this did not affect the results.
```

You can get averaged thermal conductivity by the following script.
```
from pyAF.interface import thermal_conductivity
results=thermal_conductivity('setup.yaml')
import numpy as np
kappa=np.sum(results['thermal_conductivity'])
print(kappa)
```

### example for 2-dim case
---
input for the 2D system is as follows.
```
structure_file: 'optimized.vasp'
dyn_file: 'Dyn.form'
style: 'lammps-regular'
temperature: 300
broadening_factor: 5.0
using_mean_spacing: True
omega_threshould: 10.0
broadening_threshould: 0.01
two_dim: True #two dimensional system option
vdw_thickness: 3.4  #van der Waals thickness for 2D system
symmetrize_fc: True
fix_diag: True
```

example of the script is as follows.
```
import sys
import pyAF
from pyAF.interface import resolved_thermal_conductivity
results=resolved_thermal_conductivity('setup.yaml')
import numpy as np
kappa=np.sum(results['thermal_conductivity'])
print(kappa)
```

### General tips
---
- order of LAMMPS dynamical matrix element

In parallel calculation case, the order of dynamical matrix element may be different from the order of atoms in the structure file.
(the output of `write_data` command is not sorted by atom ID)
In this case, if you convert the lammps-data file to VASP POSCAR format, 
the calculation results will be wrong. 

### modules in the interface
---
`interface.thermal_conductivity` returns the dictionary of `{'freq','diffusivity','thermal_conductivity'}` .
Each elements contain the information per mode. 

`interface.resolved_thermal_conductivity` returns the dictionary of `{'freq','diffusivity','thermal_conductivity'}`.  
In this case, diffusivity and thermal_conductivity is not averaged, thus they have three elements per mode (x,y,z components).

**The units of frequency, diffusivity, thermal conductivity are cm-1, cm^2/s, W/m K, respectively.**

## Ref
[1]Philip B. Allen and Joseph L. Feldman. Thermal conductivity of disordered harmonic solids. Phys. Rev. B, Vol. 48, pp. 12581–12588, Nov 1993.