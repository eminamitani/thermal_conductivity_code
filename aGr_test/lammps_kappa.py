import sys
import pyAF
from pyAF.interface import resolved_thermal_conductivity
results=resolved_thermal_conductivity('setup.yaml')
import numpy as np
kappa=np.sum(results['thermal_conductivity'])
print(kappa)
