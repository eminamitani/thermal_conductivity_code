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
        self.two_dim=input['two_dim']
        if self.two_dim:
            self.vdw_thickness=input['vdw_thickness']
     
def thermal_conductivity(setup_file):
    calc_setup=setup(setup_file)
    from pyAF.thermal_conductivity_AF import get_thermal_conductivity
    results=get_thermal_conductivity(calc_setup)
    return results

def resolved_thermal_conductivity(setup_file):
    calc_setup=setup(setup_file)
    from pyAF.thermal_conductivity_AF import get_resolved_thermal_conductivity
    results=get_resolved_thermal_conductivity(calc_setup)
    return results

def thermal_conductivity_THz(setup_file):
    calc_setup=setup(setup_file)
    from pyAF.thermal_conductivity_AF import get_thermal_conductivity_THz_unit
    results=get_thermal_conductivity_THz_unit(calc_setup)
    return results





