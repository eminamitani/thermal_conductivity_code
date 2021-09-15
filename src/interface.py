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
        
class physical_constants:

    def __init__(self) -> None:
        #make invisivle to act as constant
        #Boltzmanns constant (J/K)
        self.__BOLTZMANN_CONSTAT=1.38066244e-23
        #Planck's constant (Js)
        self.__PLANCK_CONSTANT=6.62617636e-34
        #Speed of light (in cm/s)
        self.__SPEED_OF_LIGHT=2.99792458e10
        #eV to J
        self.__eV_J=1.6021917e-19
        #Avogadros number
        self.__AVOGADRO=6.022045e23

        
    @property
    def scale_THz(self):
        #scale factor to make frequency THz unit
        return np.sqrt(0.1*self.__eV_J*self.__AVOGADRO)/(2.0*np.pi)
    
    @property
    def scale_cm(self):
        return np.sqrt(1.0e23*self.__eV_J*self.__AVOGADRO)/self.__SPEED_OF_LIGHT/(2.0*np.pi)
    
    @property
    def BOLTZMANN_CONSTAT(self):
        return self.__BOLTZMANN_CONSTAT
    
    @property
    def PLANCK_CONSTANT(self):
        return self.__PLANCK_CONSTANT

    @property
    def eV_J(self):
        return self.__eV_J
    
    @property
    def AVOGADRO(self):
        return self.__AVOGADRO
    
    @property
    def SPEED_OF_LIGHT(self):
        return self.__SPEED_OF_LIGHT

def thermal_conductivity(setup):
    if(setup.style=='lammps-legular'):
        pass




