import numpy as np
class physical_constants:

    def __init__(self):
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