import yaml
import numpy as np
class setup:
    def __init__(self, setup)-> None:
        input=yaml.safe_load(setup)
        self.structure_file=input['structure_file']
        self.dynfile=input['dyn_file']
        self.style=input['style']
        self.temperature=input['temperature']
        self.brodening_factor=input['brodening_factor']
        self.using_mean_spacing=input['using_mean_spacing']
        self.omega_threshould=input['omega_threshould']
        self.broadening_threshould=input['broadening_threshould']
     
def thermal_conductivity(setup_file):
    calc_setup=setup(setup_file)
    if(calc_setup.style=='lammps-legular'):
        from thermal_conductivity_AF import thermal_conductivity_lammps_legular
        results=thermal_conductivity_lammps_legular(calc_setup)




