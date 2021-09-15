import yaml
import numpy as np
class setup:
    def __init__(self, setup):
        with open(setup,'r') as obj:
            input=yaml.safe_load(obj)
        print('loaded calculation setups:')
        print(input)
        self.structure_file=input['structure_file']
        self.dyn_file=input['dyn_file']
        self.style=input['style']
        self.temperature=input['temperature']
        self.broadening_factor=input['broadening_factor']
        self.using_mean_spacing=input['using_mean_spacing']
        self.omega_threshould=input['omega_threshould']
        self.broadening_threshould=input['broadening_threshould']
     
def thermal_conductivity(setup_file):
    calc_setup=setup(setup_file)
    if(calc_setup.style=='lammps-regular'):
        from thermal_conductivity_AF import thermal_conductivity_lammps_regular
        results=thermal_conductivity_lammps_regular(calc_setup)




