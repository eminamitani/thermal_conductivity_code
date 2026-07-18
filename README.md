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
#optional mode-gate tolerances
translation_overlap_min: 0.99
translation_capture_min: 2.999
translation_leakage_max: 0.001
asr_residual_max: 1.0e-10
negative_mode_tolerance_cm: 0.1
flexural_polarization_min: 0.8
# Test-only override for a known 2D flexural failure. Keep False for production.
allow_2d_flexural_fail: False
```

You can get averaged thermal conductivity by the following script.
```
from pyAF.interface import thermal_conductivity
results=thermal_conductivity('setup.yaml')
import numpy as np
kappa=np.sum(results['thermal_conductivity'])
print(kappa)
```

The same calculation can be returned on a THz frequency axis:
```
from pyAF.interface import thermal_conductivity_THz
results = thermal_conductivity_THz('setup.yaml')
```

Both interfaces use the same YAML parameter units. `omega_threshould` is in
cm-1. When `using_mean_spacing` is `False`, `broadening_factor` is also in
cm-1; when it is `True`, `broadening_factor` is dimensionless and multiplies
the mean mode spacing. `broadening_threshould` is a Lorentzian-weight cutoff
in (cm-1)^-1. The THz implementation converts these values internally.

`broadening_threshould` corresponds to GULP's
[`lorentzian_tolerance`](https://gulp.curtin.edu.au/help/help_44_txt.html).
The sample value `0.01` is retained as the GULP-compatible default: a mode-pair
contribution is set to zero when its Lorentzian weight is below this value.
Setting `broadening_threshould: 0.0` keeps the full Lorentzian without a
drop-tolerance cutoff.

A finite value is a computational truncation of the Lorentzian tails, not a
physical broadening parameter. It can change the quantitative mode
diffusivities and thermal conductivity, so calculations using a finite
tolerance should report its value and verify convergence as the tolerance is
reduced toward zero. For the bundled aSi512 input, changing only this setting
from `0.01` to `0.0` changes the summed conductivity at 300 K from
`0.980092` to `1.127902 W/mK`; results obtained with different tolerances
should therefore not be compared as if they used the same AF numerical
definition.

### Input validation and periodic images

All three conductivity routes require a cell matrix whose off-diagonal
elements are no larger than `0.01 Angstrom`. A non-orthogonal cell stops the
calculation with `ValueError`. The check uses
`cell - np.diag(np.diag(cell))`; the earlier
`cell - np.diag(cell)` expression incorrectly broadcast a length-three vector
over the matrix.

The dynamical matrix must have exactly `(3 * natom, 3 * natom)` elements.
Malformed LAMMPS matrices and matrices returned by either conversion or
symmetrization route stop before the eigensolver or velocity-operator
calculation. `get_Sij` also rejects inconsistent velocity-operator and
eigenvector shapes.

For `two_dim: True`, use `resolved_thermal_conductivity`. The scalar cm-1 and
THz interfaces remain available for compatibility, but emit a warning because
they use the full 3D cell volume, including vacuum; their conductivity
therefore depends on the chosen area and thickness. Confirm that this is the
intended normalization before using such a result quantitatively.

The velocity-operator construction also performs a periodic-image preflight.
It warns when a non-negligible interatomic dynamical-matrix block
(larger than `1e-8` of the largest matrix element) occurs at a minimum-image
distance greater than or equal to half the shortest cell length. In that
regime, a Gamma-point aggregated dynamical matrix may combine contributions
from multiple periodic images that this implementation cannot distinguish.
Use image-resolved force constants and a corresponding velocity operator
before treating such a result as quantitative.

This is a conservative warning, not a proof that periodic-image information is
complete: once force constants have been summed into a Gamma-point matrix,
distinct image contributions cannot in general be reconstructed from the
matrix and structure alone.

### Mode-gate meanings for 3D and 2D

Before evaluating transport, pyAF identifies the three mass-weighted rigid
translations from their eigenvector overlaps. Both 3D and 2D systems have
three uniform translations (`x`, `y`, and `z`). These modes are identified by
their eigenvectors rather than by assuming that they occupy fixed mode
indices.

Gate 1 is a stability check applied to **every negative non-translation
mode**. `omega_threshould` does not cancel or hide this check.

| common Gate 1 condition | meaning |
| --- | --- |
| no negative non-translation modes | `PASS` |
| all negative non-translation modes are within `negative_mode_tolerance_cm` | numerical-error region; continue with a review warning |
| at least one negative non-translation mode exceeds the tolerance | structural-stability failure; stop |

For a 3D input, a tolerance-level warning is `REVIEW` and a significant
negative mode is `FAIL`.

For a 2D input, the same common check runs first. The out-of-plane polarization
is then used only as an additional diagnosis:

| 2D negative-mode composition | Gate 1 status |
| --- | --- |
| all small negative modes are ZA-like | `REVIEW_2D_FLEXURAL` |
| at least one small negative mode is non-ZA, with no significant mode | `REVIEW` |
| all significant negative modes are ZA-like | `FAIL_2D_FLEXURAL` |
| at least one significant negative mode is non-ZA | `FAIL` |

ZA-like and non-ZA negative modes are always listed and warned separately when
both are present. Thus, ZA classification refines the physical interpretation;
it does not restrict Gate 1 to flexural modes.

Gate 2 checks the treatment of the three rigid translations. It returns
`PASS` when the three strongest translation-overlap modes are the three lowest
modes and the overlap and acoustic-sum-rule criteria pass. If that test is not
clean, but `omega_threshould` removes the complete translation subspace with
negligible translation leakage above the cutoff, Gate 2 returns
`PASS_WITH_CUTOFF` for 3D or `PASS_WITH_CUTOFF_2D` for 2D. These are
cutoff-conditioned results: genuine physical low-frequency modes may have
been removed together with the translations. If the cutoff does not isolate
the translation subspace, Gate 2 is `REVIEW` and the calculation stops.

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
negative_mode_tolerance_cm: 0.1
flexural_polarization_min: 0.8
allow_2d_flexural_fail: False
```

example of the script is as follows.
```
from pyAF.interface import resolved_thermal_conductivity
results=resolved_thermal_conductivity('setup.yaml')
print(results['mode_gate'])
print(results['thermal_conductivity_summary'])
```

For `two_dim: True`, the additional diagnostic classifies a negative
non-translation mode as ZA-like when its out-of-plane polarization is at least
`flexural_polarization_min`. All negative non-translation modes still enter the
common Gate 1 stability check. ZA-like and non-ZA subsets are reported
independently.

The returned `mode_gate` explicitly reports the number, indices, and
frequencies of positive physical ZA-like modes excluded by
`omega_threshould`. The direction-resolved result also contains
`thermal_conductivity_summary`, with `x`, `y`, `z_diagnostic`, and
`in_plane_average = (x + y) / 2`.

`allow_2d_flexural_fail: True` is an explicit exploratory override. It applies
only when every significant negative mode is ZA-like, preserves the
`TEST_ONLY_FAIL_2D_FLEXURAL` status, and excludes negative modes while allowing
the test calculation to continue. A significant non-ZA mode always remains
`FAIL` and cannot be bypassed by this option. Do not use an overridden result
as a production-quality conductivity of a stable structure.

### General tips
---
- order of LAMMPS dynamical matrix element

In parallel calculation case, the order of dynamical matrix element may be different from the order of atoms in the structure file.
(the output of `write_data` command is not sorted by atom ID)
In this case, if you convert the lammps-data file to VASP POSCAR format by ASE read & write, the calculation results will be wrong. 

### modules in the interface
---
`interface.thermal_conductivity` returns the dictionary of `{'freq','diffusivity','thermal_conductivity'}` .
Each elements contain the information per mode. 

`interface.resolved_thermal_conductivity` returns the dictionary of `{'freq','diffusivity','thermal_conductivity'}`.  
In this case, diffusivity and thermal_conductivity is not averaged, thus they have three elements per mode (x,y,z components).

For `interface.thermal_conductivity`, the units of frequency, diffusivity, and
thermal conductivity are cm-1, cm^2/s, and W/m K, respectively. For
`interface.thermal_conductivity_THz`, the frequency is in THz; diffusivity and
thermal conductivity retain cm^2/s and W/m K.

The returned dictionary also contains `mode_gate`, with Gate 1/2 status,
translation-mode indices and overlaps, acoustic-sum-rule residual, translation
leakage above the cutoff, and the number of non-translation modes excluded by
`omega_threshould`.

See [`docs/thz_unit_audit.md`](docs/thz_unit_audit.md) for the THz-unit audit
and the `aSi512_test` regression result. See
[`docs/2d_mode_gates.md`](docs/2d_mode_gates.md) for the 2D gate contract and
the exploratory `aGr_test` result.

## Ref
[1]Philip B. Allen and Joseph L. Feldman. Thermal conductivity of disordered harmonic solids. Phys. Rev. B, Vol. 48, pp. 12581–12588, Nov 1993.
