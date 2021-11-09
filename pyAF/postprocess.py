
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

    return np.linalg.norm(np.dot(polalization_L, phase)), np.linalg.norm(np.dot(polalization_T.T, phase))

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

    return np.linalg.norm(np.dot(polalization_L, phase)), np.linalg.norm(np.dot(polalization_T.T, phase))


def dynamic_structure_factor():
    '''
    evaluate dynamic structure factor
    '''
    pass

