import sys
import pyAF
from pyAF.interface import resolved_thermal_conductivity
results=resolved_thermal_conductivity('setup.yaml')
import numpy as np
kappa_xyz=np.sum(results['thermal_conductivity'],axis=0)
kappa_in_plane=0.5*(kappa_xyz[0]+kappa_xyz[1])
print('mode_gate:',results['mode_gate'])
print('kappa_x [W/mK]:',kappa_xyz[0])
print('kappa_y [W/mK]:',kappa_xyz[1])
print('kappa_z_diagnostic [W/mK]:',kappa_xyz[2])
print('kappa_in_plane [W/mK]:',kappa_in_plane)
