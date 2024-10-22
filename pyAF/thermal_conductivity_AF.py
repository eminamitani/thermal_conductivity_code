import numpy as np
from ase.io import read
from pyAF.nearest import find_nearest_optimized

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
            #dist[i,j]=find_nearest_optimized(atoms,i,j)
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


from joblib import Parallel, delayed

# 各 (i, j) ペアに対してVijを計算する関数
def compute_Vij_chunk(i, j, atoms, Dyn):
    # 距離の計算
    dist_ij = find_nearest_optimized(atoms, i, j)

    # 各成分 (x, y, z) の行列要素を計算
    Rx = np.tile(dist_ij[0], (3, 3)) * Dyn[3*i:3*i+3, 3*j:3*j+3] * -1
    Ry = np.tile(dist_ij[1], (3, 3)) * Dyn[3*i:3*i+3, 3*j:3*j+3] * -1
    Rz = np.tile(dist_ij[2], (3, 3)) * Dyn[3*i:3*i+3, 3*j:3*j+3] * -1

    return i, j, Rx, Ry, Rz

def get_Vij_from_flat_parallel(structure_file, Dyn):
    atoms = read(structure_file, format='vasp')
    natom = len(atoms.positions)

    # 並列処理で距離計算と行列生成を行う
    results = Parallel(n_jobs=-1, backend='loky')(
        delayed(compute_Vij_chunk)(i, j, atoms, Dyn) for i in range(natom) for j in range(i)
    )

    # Vx, Vy, Vz 行列の初期化
    Vx = np.zeros((3*natom, 3*natom))
    Vy = np.zeros((3*natom, 3*natom))
    Vz = np.zeros((3*natom, 3*natom))

    # 結果を辞書に保存 (i, j) -> (Rx, Ry, Rz)
    results_dict = {}
    for i, j, Rx, Ry, Rz in results:
        results_dict[(i, j)] = (Rx, Ry, Rz)

    # Vx, Vy, Vz 行列を一貫した順序で更新
    for i in range(natom):
        for j in range(i):
            if (i, j) in results_dict:
                Rx, Ry, Rz = results_dict[(i, j)]
                Vx[3*i:3*i+3, 3*j:3*j+3] = Rx
                Vy[3*i:3*i+3, 3*j:3*j+3] = Ry
                Vz[3*i:3*i+3, 3*j:3*j+3] = Rz

                # 対称性を利用して (j, i) の成分も設定
                Vx[3*j:3*j+3, 3*i:3*i+3] = Rx.T
                Vy[3*j:3*j+3, 3*i:3*i+3] = Ry.T
                Vz[3*j:3*j+3, 3*i:3*i+3] = Rz.T

    return Vx, Vy, Vz
'''
evaluate heat flux operator matrix element.
Vx, Vy, Vz is the return of get_Vij
omega--> phonon frequency
note that eigenvector is assumed to store in column order (same as the return of numpy.linalg.eig)
'''
def get_Sij(Vx,Vy,Vz, eigenvector, omega,omega_threshould,fix_diag):

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


    inv_omega=np.zeros(nmodes)
    for i in range(nmodes):
        #tentative
        if omega[i] >omega_threshould:
            inv_omega[i]=1.0/np.sqrt(omega[i])
        else:
            inv_omega[i]=0.0


    for i in range(nmodes):
        for j in range(nmodes):
            Sijx[i,j]=EVijx[i,j]*(omega[i]+omega[j])*inv_omega[i]*inv_omega[j]
            Sijy[i,j]=EVijy[i,j]*(omega[i]+omega[j])*inv_omega[i]*inv_omega[j]
            Sijz[i,j]=EVijz[i,j]*(omega[i]+omega[j])*inv_omega[i]*inv_omega[j]
    
    #fix diagonal element
    if(fix_diag):
        for i in range(nmodes):
            Sijx[i,i]=0.0
            Sijy[i,i]=0.0
            Sijz[i,i]=0.0
    
    return Sijx, Sijy, Sijz


'''
input is class setup object
'''
def get_thermal_conductivity(setup):
    from ase.io import read
    import numpy as np
    from pyAF.constants import physical_constants
    from pyAF.data_parse import symmetrize_lammps, symmetrize_phonopy
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
        if(setup.symmetrize_fc):
            print('dynamical matrix&force constants are symmetrized')
            lammps_dyn=symmetrize_lammps(atoms,setup.dyn_file)
        else:
            lammps_dyn=np.loadtxt(setup.dyn_file).reshape((nmodes,nmodes))

    elif(setup.style=='phonopy'):
        print('style is phonopy')
        if(setup.symmetrize_fc):
            print('dynamical matrix&force constants are symmetrized')
            lammps_dyn=symmetrize_phonopy(atoms,setup.dyn_file)
        else:
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
    #extract minimum index of negative frequency
    mode_negative=0
    for i in range(nmodes):
        if eigenvalue[i] <0.0:
            val=-np.sqrt(-eigenvalue[i])*pc.scale_cm
            omega.append(val)
            mode_negative=i
        else:
            val=np.sqrt(eigenvalue[i])*pc.scale_cm
            omega.append(val)
    Sx,Sy,Sz=get_Sij(Vx,Vy,Vz,eigenvector,omega,setup.omega_threshould,setup.fix_diag)

    constant = ((1.0e-17*pc.eV_J*pc.AVOGADRO)**0.5)*(pc.scale_cm**3)
    constant = np.pi*constant/48.0

    if setup.using_mean_spacing:
        dwavg=0.0
        #not consider the negative mode contribution
        for i in range(mode_negative+1,nmodes-1):
            if omega[i] > 0.0:
                dwavg+=omega[i+1]-omega[i]
            elif omega[i+1] >0.0:
                dwavg+=omega[i+1]
        dwavg=dwavg/(len(range(mode_negative+1,nmodes-1))-1)
        print('average mode spacing:{0:8f} cm-1'.format(dwavg))
        broad=setup.broadening_factor*dwavg
    else:
        broad=setup.broadening_factor
    
    Di=np.zeros(len(omega))
    #not consider the negative mode contribution
    for i in range(mode_negative+1,nmodes):
        Di_loc = 0.0
        for j in range(mode_negative+1,nmodes):
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
    from pyAF.data_parse import symmetrize_lammps, symmetrize_phonopy
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
        if(setup.symmetrize_fc):
            print('dynamical matrix&force constants are symmetrized')
            lammps_dyn=symmetrize_lammps(atoms,setup.dyn_file)

        else:
            lammps_dyn=np.loadtxt(setup.dyn_file).reshape((nmodes,nmodes))

    elif(setup.style=='phonopy'):
        print('style is phonopy')
        if(setup.symmetrize_fc):
            print('dynamical matrix&force constants are symmetrized')
            lammps_dyn=symmetrize_phonopy(atoms,setup.dyn_file)
        else:
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
    #extract minimum index of negative frequency
    mode_negative=0
    for i in range(nmodes):
        if eigenvalue[i] <0.0:
            val=-np.sqrt(-eigenvalue[i])*pc.scale_cm
            omega.append(val)
            mode_negative=i
        else:
            val=np.sqrt(eigenvalue[i])*pc.scale_cm
            omega.append(val)
    Sx,Sy,Sz=get_Sij(Vx,Vy,Vz,eigenvector,omega,setup.omega_threshould,setup.fix_diag)

    constant = ((1.0e-17*pc.eV_J*pc.AVOGADRO)**0.5)*(pc.scale_cm**3)
    #not averaged out for x-,y-,z- dimension
    constant = np.pi*constant/16.0

    if setup.using_mean_spacing:
        dwavg=0.0
        #using only positive energy side
        for i in range(mode_negative+1,nmodes-1):
            if omega[i] > 0.0:
                dwavg+=omega[i+1]-omega[i]
            elif omega[i+1] >0.0:
                dwavg+=omega[i+1]
        dwavg=dwavg/(len(range(mode_negative+1,nmodes-1))-1)
        print('average mode spacing:{0:8f} cm-1'.format(dwavg))
        broad=setup.broadening_factor*dwavg
    else:
        broad=setup.broadening_factor
    
    #x-,y-,z-direction
    Di=np.zeros((len(omega),3))
    #using only positive energy side
    for i in range(mode_negative+1,nmodes):
        Di_loc_x = 0.0
        Di_loc_y = 0.0
        Di_loc_z = 0.0
        for j in range(mode_negative+1,nmodes):
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


'''
input is class setup object
using THz unit as frequency unit
(to compare with Shiga-san's code)
'''
def get_thermal_conductivity_THz_unit(setup):
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
    #extract minimum index of negative frequency
    #omega is angular frequency
    mode_negative=0
    for i in range(nmodes):
        if eigenvalue[i] <0.0:
            val=-np.sqrt(-eigenvalue[i])*pc.scale_THz*2.0*np.pi
            omega.append(val)
            mode_negative=i
        else:
            val=np.sqrt(eigenvalue[i])*pc.scale_THz*2.0*np.pi
            omega.append(val)
    Sx,Sy,Sz=get_Sij(Vx,Vy,Vz,eigenvector,omega,setup.omega_threshould, setup.fix_diag)

    #constant = ((1.0e-17*pc.eV_J*pc.AVOGADRO)**0.5)*(pc.scale_cm**3)
    #constant = np.pi*constant/48.0

    if setup.using_mean_spacing:
        dwavg=0.0
        #not consider the negative mode contribution
        for i in range(mode_negative+1,nmodes-1):
            if omega[i] > 0.0:
                dwavg+=omega[i+1]-omega[i]
            elif omega[i+1] >0.0:
                dwavg+=omega[i+1]
        dwavg=dwavg/(len(range(mode_negative+1,nmodes-1))-1)
        print('average mode spacing:{0:8f} 2piTHz'.format(dwavg))
        broad=setup.broadening_factor*dwavg
    else:
        broad=setup.broadening_factor

    #vol: Angstrom^3
    vol = atoms.get_volume()
    #scale Sx, Sy, Sz here
    Sx=Sx*pc.hbar/(4.0*vol)
    Sy=Sy*pc.hbar/(4.0*vol)
    Sz=Sz*pc.hbar/(4.0*vol)

    constant=np.pi*vol**2/(3.0*pc.hbar**2)

    Di=np.zeros(len(omega))
    #not consider the negative mode contribution
    for i in range(mode_negative+1,nmodes):
        Di_loc = 0.0
        for j in range(mode_negative+1,nmodes):
            if(omega[i] > setup.omega_threshould):
                dwij = (1.0/np.pi)*broad/( (omega[j] - omega[i])**2 + broad**2 )
                if(dwij > setup.broadening_threshould):
                    Di_loc = Di_loc + dwij*Sx[j,i]**2+dwij*Sy[j,i]**2+dwij*Sz[j,i]**2
        Di[i] = Di[i] + Di_loc*constant/(omega[i]**2)

    #Di=Di*1.0e-4
    kappafct = 1.0e30/vol
    #1.0e12:Hz to THz
    freqfact = pc.hbar/(2.0*pc.BOLTZMANN_CONSTANT*setup.temperature)*1.0e12
    kappa_info=np.zeros((nmodes,3))

    with open('kappa_out_THz'+setup.style,'w') as kf:
        kf.write('frequency[THz]   Diffusivity[cm^2/s]   Thermal_conductivity[W/mK] \n')
        for i in range(nmodes):
            xfreq = omega[i]*freqfact
            expfreq = np.exp(xfreq)
            cv_i = pc.BOLTZMANN_CONSTANT*xfreq*xfreq*expfreq/(expfreq - 1.0)**2
            kappa_info[i]=[omega[i]/2.0/np.pi,Di[i]*1.0e4,cv_i*kappafct*Di[i]]
            kf.write('{0:8f}  {1:12f}  {2:12f}\n'.format(omega[i]/2.0/np.pi,Di[i]*1.0e4,cv_i*kappafct*Di[i]))

    return {'freq':kappa_info[:,0],'diffusivity':kappa_info[:,1],'thermal_conductivity':kappa_info[:,2]}
