
import numpy as np

def polar_coord_sampling(n_sample,diameter):
    '''
    sampling of on isosurface of wavevector for
    stastistic average in Fourier component calculation 
    '''
    theta=np.linspace(0,np.pi,n_sample,endpoint=False)
    phi=np.linspace(0,2.0*np.pi,n_sample,endpoint=False)
    x=np.array([[diameter*np.sin(t)*np.cos(p) for t in theta] for p in phi]).flatten()
    y=np.array([[diameter*np.sin(t)*np.sin(p) for t in theta] for p in phi]).flatten()
    z=np.array([[diameter*np.cos(t) for t in theta] for p in phi]).flatten()

    return np.stack([x,y,z],axis=1)

def get_Ci_ver1(eigenvector, position, k_vector):
    '''
    evaluate Fourier component of a single mode.
    ver1 specify transverse vector explicitly.
    input:
    eigenvector: np.array(3*natom), eigenvector of the mode
    position: np.array(natom,3), cartesian coordinate of the system
    k_vector: np.array(3), wavevector
    '''

    natom=len(position)
    unit_vector=k_vector/np.linalg.norm(k_vector)
    unit_vector_T=np.array([unit_vector[1],-unit_vector[0],0])
    disp=np.reshape(eigenvector,(natom,3))

    polalization_L=np.dot(unit_vector, disp.T)
    polalization_T=np.dot(unit_vector_T, disp.T)

    phase=np.exp(1.0j*np.dot(k_vector, position.T))

    return np.linalg.norm(np.dot(polalization_L, phase))**2, np.linalg.norm(np.dot(polalization_T.T, phase))**2

def get_Ci_ver2(eigenvector, position, k_vector):
    '''
    evaluate Fourier component of a single mode.
    ver2 use cross product to reproduce inner product with unit vector orthogonal to wavevector.
    input:
    eigenvector: np.array(3*natom), eigenvector of the mode
    position: np.array(natom,3), cartesian coordinate of the system
    k_vector: np.array(3), wavevector
    '''    

    natom=len(position)
    unit_vector=k_vector/np.linalg.norm(k_vector)
    disp=np.reshape(eigenvector,(natom,3))

    polalization_L=np.dot(unit_vector, disp.T)
    
    array_unit_vector=np.repeat(unit_vector.reshape(1,3),natom,axis=0)
    polalization_T=np.cross(array_unit_vector,disp) 
    phase=np.exp(1.0j*np.dot(k_vector, position.T))

    return np.linalg.norm(np.dot(polalization_L, phase))**2, np.linalg.norm(np.dot(polalization_T.T, phase))**2

def get_Ci_ver3(eigenvector, position, k_value):
    '''
    evaluate Fourier component of a single mode.
    ver3 consider the average value along [100],[010],[001] direction
    input:
    eigenvector: np.array(3*natom), eigenvector of the mode
    position: np.array(natom,3), cartesian coordinate of the system
    k_value: float, value of x-component in k-vector
    '''    

    natom=len(position)
    k_vector_x=np.array([k_value,0.0,0.0])
    k_vector_y=np.array([0.0,k_value,0.0])
    k_vector_z=np.array([0.0,0.0,k_value])

    unit_vector_x=np.array([1.0,0.0,0.0])
    unit_vector_y=np.array([0.0,1.0,0.0])
    unit_vector_z=np.array([0.0,0.0,1.0])

    disp=np.reshape(eigenvector,(natom,3))

    polalization_Lx=np.dot(unit_vector_x, disp.T)
    polalization_Ly=np.dot(unit_vector_y, disp.T)
    polalization_Lz=np.dot(unit_vector_z, disp.T)
    
    array_unit_vector_x=np.repeat(unit_vector_x.reshape(1,3),natom,axis=0)
    array_unit_vector_y=np.repeat(unit_vector_y.reshape(1,3),natom,axis=0)
    array_unit_vector_z=np.repeat(unit_vector_z.reshape(1,3),natom,axis=0)

    polalization_Tx=np.cross(array_unit_vector_x,disp) 
    polalization_Ty=np.cross(array_unit_vector_y,disp) 
    polalization_Tz=np.cross(array_unit_vector_z,disp) 

    phase_x=np.exp(1.0j*np.dot(k_vector_x, position.T))
    phase_y=np.exp(1.0j*np.dot(k_vector_y, position.T))
    phase_z=np.exp(1.0j*np.dot(k_vector_z, position.T))
    
    cl=(np.linalg.norm(np.dot(polalization_Lx,phase_x))**2+np.linalg.norm(np.dot(polalization_Ly,phase_y))**2+np.linalg.norm(np.dot(polalization_Lz,phase_z))**2)/3
    ct=(np.linalg.norm(np.dot(polalization_Tx.T,phase_x))**2+np.linalg.norm(np.dot(polalization_Ty.T,phase_y))**2+np.linalg.norm(np.dot(polalization_Tz.T,phase_z))**2)/3

    return cl,ct

def broadening(omega1, omega2, sigma):
    delta=1.0/np.pi*sigma/((omega1-omega2)**2+sigma*sigma)
    return delta

def dynamic_structure_factor(C_L, C_T, mesh_energy, frequency, smearing):
    '''
    evaluate dynamic structure factor
    C_L: np.array(n_kpt, n_modes), Fourier component for each mode
    C_T: np.array(n_kpt, n_modes), Fourier component for each mode
    mesh_energy: mesh of energy to evaluate dynamic structure factor
    frequency: vibrational eigenvalues in the system
    smearing: smearing factor
    '''
    
    n_kpt=C_L.shape[0]
    dsf_L=np.zeros((n_kpt,len(mesh_energy)))
    dsf_T=np.zeros((n_kpt,len(mesh_energy)))

    for ik in range(n_kpt):
        for ie, ene in enumerate(mesh_energy):
            for imode, mode in enumerate(frequency):
                dsf_L[ik,ie]=dsf_L[ik,ie]+C_L[ik,imode]*broadening(ene,mode,smearing)
                dsf_T[ik,ie]=dsf_T[ik,ie]+C_T[ik,imode]*broadening(ene,mode,smearing)

    return dsf_L, dsf_T

def get_IPR(eigenvector):
    '''
    Inversion perticipation ratio
    Very direct calculation (maybe there is more faster way)
    '''
    nmodes=len(eigenvector)
    ipr_result=np.zeros(nmodes)

    for i in range(nmodes):
        vector=eigenvector[:,i]
        inner=vector*vector
        tmp=inner*inner
        ipr_result[i]=tmp.sum()


    return ipr_result

def get_IPR_rev(eigenvector):
    '''
    Inversion perticipation ratio
    Very direct calculation (maybe there is more faster way)
    '''
    nmodes=len(eigenvector)
    ipr_result=np.zeros(nmodes)

    for i in range(nmodes):
        vector=eigenvector[:,i]
        ipr_result[i]=np.power(vector,4).sum()

    return ipr_result


class SQE:
    '''
    Class for calculating dynamic structure factor
    Definition: D. L. Price and J. M. Carpenter, 'Scattering function of vitreous silica,' 
    Journal of Non-Crystalline Solids, vol. 92, no. 1, pp. 153-174, 1987
    '''
    def __init__(self, eigenvectors, eigenvalues, atoms):
        '''
        eigenvectors: eigenvectors of the system, shape=(3*Natoms,3*Natoms)

        Here I assume that eigenvector is that obtained from np.linalg.eigh
        For example, 
        ```
        df_dm=pd.read_table('Dyn.form', delimiter=' ', header=None) 
        Natom=1050
        dm=np.array(df_dm).reshape(Natom*3, Natom*3)
        eigenvalues, eigenvectors=np.linalg.eigh(dm)
        ```
        eigenvalues: eigenvalues of the system, shape=(nmodes)
        atoms: atoms object from ase
        '''
        self.eigenvectors=eigenvectors
        self.eigenvalues=eigenvalues
        self.atoms=atoms
        self.positions=atoms.get_positions()
        self.masses=atoms.get_masses()
        
        #just inplemented for silica
        self.b_factor=np.ones(len(atoms))

        for i,atom in enumerate(atoms):
            if atom.symbol == 'O':
                self.b_factor[i]=5.805
            elif atom.symbol == 'Si':
                self.b_factor[i]=4.149
            elif atom.symbol == 'Na':
                self.b_factor[i]=3.63
            elif atom.symbol == 'Li':
                self.b_factor[i]=-1.90
            else:
                print('atom type {} is not implemented yet'.format(atom.symbol))

        SPEED_OF_LIGHT=2.99792458e10
        #eV to J
        eV_J=1.6021917e-19
        #Avogadros number
        AVOGADRO=6.022045e23
        planck=6.62607015e-34
        scale_THz=np.sqrt(0.1*eV_J*AVOGADRO)/(2.0*np.pi)
        Hz_eV=planck/eV_J
        THz_Hz=1.0e12
        scale_eV=scale_THz*THz_Hz*Hz_eV
        freq=np.sqrt(np.abs(eigenvalues))*np.sign(eigenvalues)
        
        #unit of energy: eV
        self.frequency=freq*scale_eV
        self.normal_modes=np.reshape(eigenvectors.T,(3*len(atoms),len(atoms),3))

        self.b2_avg=np.sum(self.b_factor**2)/len(self.b_factor)
        
        

    def get_factor(self,Q, threshould):
        '''
        normal_mode: normal mode of the system, shape=(nmodes,Natoms,3)
        frequency: frequency of the normal mode, shape=(nmodes)
        positions: positions of the atoms, shape=(Natoms,3)
        masses: masses of the atoms, shape=(Natoms) 
        b: b factor of the atoms, shape=(Natoms)
        Q: wave vector, shape=(3)
        '''
        
        phase=np.exp(-1j*(np.dot(self.positions,Q)))
        factors=np.zeros(len(self.frequency),dtype=np.complex128)

        for i,mode in enumerate(self.frequency):
            if mode>threshould:
                poralization=np.dot(self.normal_modes[i,:,:],Q)
                denominator=np.sqrt(2.0*self.masses*mode)
                atomic_factor=self.b_factor*poralization/denominator*phase

                sum_factor=np.sum(atomic_factor)
                abs=sum_factor*sum_factor.conjugate()
                factors[i]=abs
        
        return factors
    
    def mode_sum(self,Q,threshould, energy,T, gaussian_sigma):
        kb=8.617333e-05
        factors=self.get_factor(Q,threshould)
        gaussian=np.exp(-0.5*((self.frequency-energy)/gaussian_sigma)**2)
        bose_distribution=1/(np.exp(energy/(kb*T))-1)+1
        normalize_factor=bose_distribution/len(self.atoms)

        planck=6.62607015e-34
        eV_J=1.6021917e-19
        planck_eV=planck/eV_J
        AVOGADRO=6.022045e23
        unit_convert=planck_eV*planck*1.0e20*1000*AVOGADRO

        return np.sum(factors*gaussian*normalize_factor)*unit_convert/self.b2_avg/len(self.atoms)