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
        self.symmetrize_fc=input['symmetrize_fc']
        self.fix_diag=input['fix_diag']
        self.translation_overlap_min=input.get('translation_overlap_min',0.99)
        self.translation_capture_min=input.get('translation_capture_min',2.999)
        self.translation_leakage_max=input.get('translation_leakage_max',0.001)
        self.asr_residual_max=input.get('asr_residual_max',1.0e-10)
        self.negative_mode_tolerance_cm=input.get('negative_mode_tolerance_cm',0.1)
        self.flexural_polarization_min=input.get('flexural_polarization_min',0.8)
        self.allow_2d_flexural_fail=input.get('allow_2d_flexural_fail',False)
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



