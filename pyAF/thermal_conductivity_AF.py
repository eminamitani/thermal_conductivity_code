import numpy as np
from ase.io import read

'''
evaluate velocity operator.
structure file: unitcell structure with vasp POSCAR format
Dyn: Flat format (low:0x,0y,0z....., column:0x,0y,0z....) natom*3xnatom*3 Dynamical matrix
(already scaled by mass, regular output of lammps dynamical_matrix) 
'''
#using faster ortholombic algorithm
def get_Vij_from_flat(structure_file,Dyn):
    atoms=read(structure_file,format='vasp')
    natom=len(atoms.positions)

    dist=np.zeros((natom,natom,3))

    from pyAF.nearest import find_nearest_ortho
    dist=np.zeros((natom,natom,3))
    positions=atoms.positions
    cell=atoms.cell
    for i in range(natom):  
        for j in range(i):
            dist[i,j]=find_nearest_ortho(positions,cell,i,j)
            #invert
            dist[j,i]=-dist[i,j]
    
    
    Rx=np.repeat(dist[:,:,0],3,axis=1)
    Rx=np.repeat(Rx,3,axis=0)
    Ry=np.repeat(dist[:,:,1],3,axis=1)
    Ry=np.repeat(Ry,3,axis=0)
    Rz=np.repeat(dist[:,:,2],3,axis=1)
    Rz=np.repeat(Rz,3,axis=0)  

    #Hadamard product
    Vx=Rx*Dyn*-1
    Vy=Ry*Dyn*-1
    Vz=Rz*Dyn*-1

    return Vx, Vy, Vz

'''
evaluate velocity operator. Old version.
structure file: unitcell structure with vasp POSCAR format
FC_file: force constant file of phonopy format 
'''
def get_Vij(structure_file,FC_file):
    atoms=read(structure_file,format='vasp')
    masses=atoms.get_masses()
    natom=len(atoms.positions)

    dist=np.zeros((natom,natom,3))

    from pyAF.nearest import find_nearest
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


'''
input is class setup object
'''
def get_thermal_conductivity(setup):
    from ase.io import read
    import numpy as np
    from pyAF.constants import physical_constants
    print('enter thermal conductivity calculation')
    structure_file=setup.structure_file
    atoms=read(structure_file,format='vasp')
    natom=len(atoms.positions)
    cell=atoms.cell
    '''
    this code assume orthorombic cell, check
    '''
    celldiag=np.diag(cell)
    cellnondiag=cell-celldiag
    if(np.sum(cellnondiag) > 0.01):
        assert 'cell vector has nondiagonal element. This code only support orthorhombic system. please check!'
    
    positions=atoms.positions
    masses=atoms.get_masses()
    nmodes=natom*3
    '''
    this module returns average of x-,y-,z- direction.
    The volume to scale the thermal conductivity is the volume of cell.
    For 2D system, resolved version is better to use, thus, here check and assert 
    '''
    if(setup.two_dim):
        assert "for 2D system, use resolved version is better. \
            In resolved version, x-,y-,z- direction outputted separetely and you can set vdw_thickness to set volume"
            
    if(setup.style=='lammps-regular'):
        print('style is lammps-regular')
        lammps_dyn=np.loadtxt(setup.dyn_file).reshape((nmodes,nmodes))

    elif(setup.style=='phonopy'):
        print('style is phonopy')
        #convert phonopy style force constant to mass scaled lammps format dynamical matrix
        from pyAF.data_parse import read_fc_phonopy,phonopy_to_flat
        fc_scaled=read_fc_phonopy(setup.dyn_file,natom, masses)
        lammps_dyn=phonopy_to_flat(fc_scaled,natom)
    else:
        print('not supported style')
        return

    Vx,Vy,Vz=get_Vij_from_flat(structure_file,lammps_dyn)
    eigenvalue, eigenvector=np.linalg.eigh(lammps_dyn)
    pc=physical_constants()
    omega=[]
    for i in range(nmodes):
        if eigenvalue[i] <0.0:
            val=np.sqrt(-eigenvalue[i])*pc.scale_cm
            omega.append(val)
        else:
            val=np.sqrt(eigenvalue[i])*pc.scale_cm
            omega.append(val)
    Sx,Sy,Sz=get_Sij(Vx,Vy,Vz,eigenvector,omega)

    constant = ((1.0e-17*pc.eV_J*pc.AVOGADRO)**0.5)*(pc.scale_cm**3)
    constant = np.pi*constant/48.0

    if setup.using_mean_spacing:
        dwavg=0.0
        for i in range(len(omega)-1):
            if omega[i] > 0.0:
                dwavg+=omega[i+1]-omega[i]
            elif omega[i+1] >0.0:
                dwavg+=omega[i+1]
        dwavg=dwavg/(len(omega)-1)
        broad=setup.broadening_factor*dwavg
    else:
        broad=setup.broadening_factor
    
    Di=np.zeros(len(omega))
    for i in range(nmodes):
        Di_loc = 0.0
        for j in range(nmodes):
            if(omega[i] > setup.omega_threshould):
                dwij = (1.0/np.pi)*broad/( (omega[j] - omega[i])**2 + broad**2 )
                if(dwij > setup.broadening_threshould):
                    Di_loc = Di_loc + dwij*Sx[j,i]**2+dwij*Sy[j,i]**2+dwij*Sz[j,i]**2
        Di[i] = Di[i] + Di_loc*constant/(omega[i]**2)

    vol = atoms.get_volume()
    kappafct = 1.0e30/vol
    cmfact = pc.PLANCK_CONSTANT*pc.SPEED_OF_LIGHT/(pc.BOLTZMANN_CONSTANT*setup.temperature)
    kappa_info=np.zeros((nmodes,3))

    with open('kappa_out_'+setup.style,'w') as kf:
        kf.write('frequency[cm-1]   Diffusivity[cm^2/s]   Thermal_conductivity[W/mK] \n')
        for i in range(nmodes):
            xfreq = omega[i]*cmfact
            expfreq = np.exp(xfreq)
            cv_i = pc.BOLTZMANN_CONSTANT*xfreq*xfreq*expfreq/(expfreq - 1.0)**2
            kappa_info[i]=[omega[i],Di[i]*1.0e4,cv_i*kappafct*Di[i]]
            kf.write('{0:8f}  {1:12f}  {2:12f}\n'.format(omega[i],Di[i]*1.0e4,cv_i*kappafct*Di[i]))

    return {'freq':kappa_info[:,0],'diffusivity':kappa_info[:,1],'thermal_conductivity':kappa_info[:,2]}

'''
input is class setup object.
thermal conductivity for x,y,z direction is outputted without taking average
'''
def get_resolved_thermal_conductivity(setup):
    from ase.io import read
    import numpy as np
    from pyAF.constants import physical_constants
    print('enter thermal conductivity calculation')
    structure_file=setup.structure_file
    atoms=read(structure_file,format='vasp')
    natom=len(atoms.positions)
    cell=atoms.cell
    '''
    this code assume orthorombic cell, check
    '''
    celldiag=np.diag(cell)
    cellnondiag=cell-celldiag
    if(np.sum(cellnondiag) > 0.01):
        assert 'cell vector has nondiagonal element. This code only support orthorhombic system. please check!'

    positions=atoms.positions
    masses=atoms.get_masses()
    nmodes=natom*3
    if(setup.style=='lammps-regular'):
        print('style is lammps-regular')
        lammps_dyn=np.loadtxt(setup.dyn_file).reshape((nmodes,nmodes))

    elif(setup.style=='phonopy'):
        print('style is phonopy')
        #convert phonopy style force constant to mass scaled lammps format dynamical matrix
        from pyAF.data_parse import read_fc_phonopy,phonopy_to_flat
        fc_scaled=read_fc_phonopy(setup.dyn_file,natom, masses)
        lammps_dyn=phonopy_to_flat(fc_scaled,natom)
    else:
        print('not supported style')
        return

    Vx,Vy,Vz=get_Vij_from_flat(structure_file,lammps_dyn)
    eigenvalue, eigenvector=np.linalg.eigh(lammps_dyn)
    pc=physical_constants()
    omega=[]
    for i in range(nmodes):
        if eigenvalue[i] <0.0:
            val=np.sqrt(-eigenvalue[i])*pc.scale_cm
            omega.append(val)
        else:
            val=np.sqrt(eigenvalue[i])*pc.scale_cm
            omega.append(val)
    Sx,Sy,Sz=get_Sij(Vx,Vy,Vz,eigenvector,omega)

    constant = ((1.0e-17*pc.eV_J*pc.AVOGADRO)**0.5)*(pc.scale_cm**3)
    #not averaged out for x-,y-,z- dimension
    constant = np.pi*constant/16.0

    if setup.using_mean_spacing:
        dwavg=0.0
        for i in range(len(omega)-1):
            if omega[i] > 0.0:
                dwavg+=omega[i+1]-omega[i]
            elif omega[i+1] >0.0:
                dwavg+=omega[i+1]
        dwavg=dwavg/(len(omega)-1)
        broad=setup.broadening_factor*dwavg
    else:
        broad=setup.broadening_factor
    
    #x-,y-,z-direction
    Di=np.zeros((len(omega),3))
    for i in range(nmodes):
        Di_loc_x = 0.0
        Di_loc_y = 0.0
        Di_loc_z = 0.0
        for j in range(nmodes):
            if(omega[i] > setup.omega_threshould):
                dwij = (1.0/np.pi)*broad/( (omega[j] - omega[i])**2 + broad**2 )
                if(dwij > setup.broadening_threshould):
                    Di_loc_x += dwij*Sx[j,i]**2
                    Di_loc_y += dwij*Sy[j,i]**2
                    Di_loc_z += dwij*Sz[j,i]**2

        Di[i,0] += Di_loc_x*constant/(omega[i]**2)
        Di[i,1] += Di_loc_y*constant/(omega[i]**2)
        Di[i,2] += Di_loc_z*constant/(omega[i]**2)
    if(setup.two_dim):
        vol=atoms.cell[0,0]*atoms.cell[1,1]*setup.vdw_thickness
    else:
        vol = atoms.get_volume()

    kappafct = 1.0e30/vol
    cmfact = pc.PLANCK_CONSTANT*pc.SPEED_OF_LIGHT/(pc.BOLTZMANN_CONSTANT*setup.temperature)
    
    diffusivity=np.zeros((nmodes,3))
    kappa=np.zeros((nmodes,3))

    with open('kappa_out_'+setup.style,'w') as kf:
        kf.write('frequency[cm-1]   Diffusivity[cm^2/s]: x,y,z   Thermal_conductivity[W/mK]: x,y,z \n')
        for i in range(nmodes):
            xfreq = omega[i]*cmfact
            expfreq = np.exp(xfreq)
            cv_i = pc.BOLTZMANN_CONSTANT*xfreq*xfreq*expfreq/(expfreq - 1.0)**2
            diffusivity[i]=Di[i]*1.0e4
            kappa[i]=cv_i*kappafct*Di[i]
            kf.write('{0:8f}  {1:8f}  {2:8f} {3:8f} {4:8f} {5:8f} {6:8f}　\n'.
            format(omega[i],diffusivity[i,0],diffusivity[i,1],diffusivity[i,2],
            kappa[i,0],kappa[i,1],kappa[i,2]))

    return {'freq':omega,'diffusivity':diffusivity,'thermal_conductivity':kappa}



