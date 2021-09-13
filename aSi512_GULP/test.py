import os
import sys
if __package__ is None:
    module_path = os.path.abspath(os.path.join('..'))
    if module_path not in sys.path:
        sys.path.append(module_path)
    __package__= "myparent"

from src import thermal_conductivity_AF
structure_file='optimized.vasp'
FC_file='FORCE_CONSTANTS_2ND'

Vx, Vy, Vz=thermal_conductivity_AF.get_Vij(structure_file,FC_file)