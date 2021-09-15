import sys
import pyAF
from pyAF.interface import thermal_conductivity
results=thermal_conductivity('setup.yaml')
import numpy as np
kappa=np.sum(results['thermal_conductivity'])
print(kappa)
