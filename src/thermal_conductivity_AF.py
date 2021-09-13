import numpy as np
from ase.io import read

'''
evaluate velocity operator.
structure file: unitcell structure with vasp POSCAR format
FC_file: force constant file of phonopy format 
'''
def get_Vij(structure_file,FC_file):
    atoms=read(structure_file,format='vasp')
    masses=atoms.get_masses()
    natom=len(atoms.positions)

    dist=np.zeros((natom,natom,3))

    from .nearest import find_nearest
    for i in range(natom):  
        for j in range(i):
            xdc,ydc,zdc,rmin=find_nearest(atoms,i,j)
            dist[i,j]=np.array([xdc,ydc,zdc])
    
    #reading phonopy format force constant
    with open(FC_file,'r') as fc:
        lines=fc.readlines()

        nlineblock=4
        fc_all=np.zeros((natom,natom,3,3))
        start=1
        for i in range(natom):
            for j in range(natom):
                fc_block=lines[start+1:start+nlineblock]
                fc=np.loadtxt(fc_block)
                fc_all[i,j]=fc/np.sqrt(masses[i]*masses[j])
                #fc_all[i,j]=fc
                start=start+nlineblock
    
    Rx=dist[:,:,0]
    Ry=dist[:,:,1]
    Rz=dist[:,:,2]  

    Vx=np.zeros((natom,natom,3,3))
    Vy=np.zeros((natom,natom,3,3))
    Vz=np.zeros((natom,natom,3,3))

    #loop only for lower triangle
    #make sure that Vij=-Vji
    for i in range(natom):
        for j in range(i):
            Vijx=-Rx[i,j]*fc_all[i,j]
            Vx[i,j]=Vijx
            Vx[j,i]=-Vijx.T

            Vijy=-Ry[i,j]*fc_all[i,j]
            Vy[i,j]=Vijy
            Vy[j,i]=-Vijy.T

            Vijz=-Rz[i,j]*fc_all[i,j]
            Vz[i,j]=Vijz
            Vz[j,i]=-Vijz.T

    #reshape to (natom*3,natom*3) that matches with GULP format        
    flatVx=np.reshape(Vx.transpose(0,2,1,3),(natom*3,natom*3))
    flatVy=np.reshape(Vy.transpose(0,2,1,3),(natom*3,natom*3))
    flatVz=np.reshape(Vz.transpose(0,2,1,3),(natom*3,natom*3))

    return flatVx, flatVy, flatVz

'''
evaluate heat flux operator matrix element.
Vx, Vy, Vz is the return of get_Vij
omega--> phonon frequency
note that eigenvector is assumed to store in column order (same as the return of numpy.linalg.eig)
'''
def get_Sij(Vx,Vy,Vz, eigenvector, omega):

    nmodes=len(omega)

    #confirm matrix shape
    if(Vx.shape[0]!=Vx.shape[1] or Vx.shape[0]!=nmodes):
        assert "matrix shape Vx is strange"

    if(Vy.shape[0]!=Vy.shape[1] or Vy.shape[0]!=nmodes):
        assert "matrix shape Vy is strange"   

    if(Vz.shape[0]!=Vz.shape[1] or Vz.shape[0]!=nmodes):
        assert "matrix shape Vz is strange"  

    if(eigenvector.shape[0]!=eigenvector.shape[1] or eigenvector.shape[0]!=nmodes):
        assert "matrix shape eigenvector is strange" 

    #here the shape of eigenvector is assumed to that the typical return of numpy.linalg.eig
    #thus, each column is the eigenvector of each mode
    tmpx=np.dot(Vx,eigenvector)
    EVijx=np.dot(eigenvector.T,tmpx)

    tmpy=np.dot(Vy,eigenvector)
    EVijy=np.dot(eigenvector.T,tmpy)

    tmpz=np.dot(Vz,eigenvector)
    EVijz=np.dot(eigenvector.T,tmpz)


    Sijx=np.zeros((nmodes,nmodes))
    Sijy=np.zeros((nmodes,nmodes))
    Sijz=np.zeros((nmodes,nmodes))

    nfreqmin = 0
    inv_omega=np.zeros(nmodes)
    for i in range(nmodes):
        if (nfreqmin==0 and omega[i]> 0.01):
            nfreqmin = i
        if omega[i] >0.0:
            inv_omega[i]=1.0/np.sqrt(omega[i])
        else:
            inv_omega[i]=0.0


    for i in range(nmodes):
        for j in range(nmodes):
            Sijx[i,j]=EVijx[i,j]*(omega[i]+omega[j])*inv_omega[i]*inv_omega[j]
            Sijy[i,j]=EVijy[i,j]*(omega[i]+omega[j])*inv_omega[i]*inv_omega[j]
            Sijz[i,j]=EVijz[i,j]*(omega[i]+omega[j])*inv_omega[i]*inv_omega[j]
    
    return Sijx, Sijy, Sijz