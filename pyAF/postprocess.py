
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

    return np.linalg.norm(np.dot(polalization_L, phase))**2, np.linalg.norm(np.dot(polalization_T.T, phase))**2/2

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
