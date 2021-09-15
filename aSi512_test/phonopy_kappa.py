from pyAF.interface import thermal_conductivity
results=thermal_conductivity('setup_phonopy.yaml')
import numpy as np
np.sum(results['thermal_conductivity'])