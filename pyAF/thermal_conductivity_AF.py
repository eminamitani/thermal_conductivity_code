import numpy as np
from ase.io import read
from pyAF.nearest import find_nearest_optimized


def _mode_heat_capacity(xfreq, boltzmann_constant):
    """Return the harmonic-mode heat capacity with a stable zero-frequency limit."""
    xfreq = abs(float(xfreq))
    if xfreq < 1.0e-8:
        return boltzmann_constant
    if xfreq > 700.0:
        return 0.0
    denominator = -np.expm1(-xfreq)
    return (
        boltzmann_constant
        * xfreq
        * xfreq
        * np.exp(-xfreq)
        / denominator**2
    )


def _angular_thz_per_wavenumber(pc):
    """Conversion factor from cm^-1 to angular frequency in 2*pi*THz."""
    return pc.scale_THz * 2.0 * np.pi / pc.scale_cm


def _translation_mode_overlaps(eigenvectors, masses):
    """Return each mass-weighted eigenvector's overlap with rigid translations."""
    nmodes = eigenvectors.shape[0]
    natom = len(masses)
    if nmodes != 3 * natom:
        raise ValueError("eigenvector and mass dimensions are inconsistent")

    translations = np.zeros((nmodes, 3))
    sqrt_masses = np.sqrt(masses)
    for axis in range(3):
        translations[axis::3, axis] = sqrt_masses
    translations /= np.linalg.norm(translations, axis=0)
    overlaps = np.sum((eigenvectors.T @ translations) ** 2, axis=1)
    return overlaps, translations


def _evaluate_mode_gates(
    setup,
    eigenvalues,
    eigenvectors,
    masses,
    frequencies_cm,
):
    """Evaluate stability and rigid-translation gates before AF transport."""
    overlap_min = getattr(setup, "translation_overlap_min", 0.99)
    capture_min = getattr(setup, "translation_capture_min", 2.999)
    leakage_max = getattr(setup, "translation_leakage_max", 0.001)
    asr_residual_max = getattr(setup, "asr_residual_max", 1.0e-10)
    negative_tolerance = getattr(setup, "negative_mode_tolerance_cm", 0.1)
    two_dim = bool(getattr(setup, "two_dim", False))
    flexural_min = getattr(setup, "flexural_polarization_min", 0.8)
    allow_2d_flexural_fail = bool(
        getattr(setup, "allow_2d_flexural_fail", False)
    )

    frequencies_cm = np.asarray(frequencies_cm)
    overlaps, translations = _translation_mode_overlaps(eigenvectors, masses)
    translation_indices = np.argsort(overlaps)[-3:][::-1]
    translation_mask = np.zeros(len(eigenvalues), dtype=bool)
    translation_mask[translation_indices] = True
    lowest_indices = np.argsort(eigenvalues)[:3]

    translation_capture = float(np.sum(overlaps[translation_indices]))
    minimum_translation_overlap = float(
        np.min(overlaps[translation_indices])
    )
    dynamical_norm = np.linalg.norm(eigenvalues)
    projected_translation = eigenvectors.T @ translations
    asr_residual = eigenvectors @ (
        eigenvalues[:, None] * projected_translation
    )
    if dynamical_norm == 0.0:
        asr_residual_ratio = 0.0
    else:
        asr_residual_ratio = float(
            np.linalg.norm(asr_residual) / dynamical_norm
        )

    nmodes = len(eigenvalues)
    mode_vectors = eigenvectors.reshape((len(masses), 3, nmodes))
    z_polarization = np.sum(mode_vectors[:, 2, :] ** 2, axis=0)
    nontranslation_mask = ~translation_mask
    negative_nontranslation = np.flatnonzero(
        nontranslation_mask & (frequencies_cm < 0.0)
    )
    unstable_nontranslation = np.flatnonzero(
        nontranslation_mask
        & (frequencies_cm < -negative_tolerance)
    )
    small_negative_nontranslation = np.setdiff1d(
        negative_nontranslation,
        unstable_nontranslation,
        assume_unique=True,
    )

    negative_flexural = np.array([], dtype=int)
    if two_dim:
        negative_flexural = negative_nontranslation[
            z_polarization[negative_nontranslation] >= flexural_min
        ]
    negative_other = np.setdiff1d(
        negative_nontranslation,
        negative_flexural,
        assume_unique=True,
    )
    unstable_flexural = np.intersect1d(
        unstable_nontranslation,
        negative_flexural,
        assume_unique=True,
    )
    small_negative_flexural = np.intersect1d(
        small_negative_nontranslation,
        negative_flexural,
        assume_unique=True,
    )
    unstable_other = np.setdiff1d(
        unstable_nontranslation,
        unstable_flexural,
        assume_unique=True,
    )
    small_negative_other = np.setdiff1d(
        small_negative_nontranslation,
        small_negative_flexural,
        assume_unique=True,
    )

    if len(unstable_other) > 0:
        gate1_status = "FAIL"
    elif len(unstable_flexural) > 0:
        gate1_status = "FAIL_2D_FLEXURAL"
    elif len(small_negative_other) > 0:
        gate1_status = "REVIEW"
    elif len(small_negative_flexural) > 0:
        gate1_status = "REVIEW_2D_FLEXURAL"
    else:
        gate1_status = "PASS"

    ordering_ok = set(translation_indices) == set(lowest_indices)
    overlap_ok = (
        minimum_translation_overlap >= overlap_min
        and translation_capture >= capture_min
    )
    asr_ok = asr_residual_ratio <= asr_residual_max

    translation_below_cutoff = (
        translation_mask
        & (np.abs(frequencies_cm) <= setup.omega_threshould)
    )
    positive_physical_below_cutoff = (
        nontranslation_mask
        & (frequencies_cm > 0.0)
        & (frequencies_cm <= setup.omega_threshould)
    )
    cutoff_inactive = (
        translation_below_cutoff
        | positive_physical_below_cutoff
        | (nontranslation_mask & (frequencies_cm < 0.0))
    )
    cutoff_active = frequencies_cm > setup.omega_threshould
    inactive_translation_capture = float(
        np.sum(overlaps[cutoff_inactive])
    )
    active_translation_leakage = float(
        np.sum(overlaps[cutoff_active])
    )
    cutoff_protects = (
        inactive_translation_capture >= capture_min
        and active_translation_leakage <= leakage_max
    )

    warnings = []
    if ordering_ok and overlap_ok and asr_ok:
        gate2_status = "PASS"
    elif cutoff_protects:
        if two_dim:
            gate2_status = "PASS_WITH_CUTOFF_2D"
        else:
            gate2_status = "PASS_WITH_CUTOFF"
        warnings.append(
            "Gate 2 did not uniquely validate the three lowest modes; "
            "the configured omega_threshould removes the full translation "
            "subspace and may also remove physical low-frequency modes."
        )
    else:
        gate2_status = "REVIEW"
        warnings.append(
            "Gate 2 requires review: rigid translations are not cleanly "
            "identified as the three lowest modes and translation character "
            "leaks above omega_threshould."
        )

    if len(negative_nontranslation) > 0:
        warnings.append(
            "Gate 1 found "
            f"{len(negative_nontranslation)} negative non-translation "
            f"mode(s): {len(unstable_nontranslation)} beyond and "
            f"{len(small_negative_nontranslation)} within "
            "negative_mode_tolerance_cm."
        )
    if len(unstable_nontranslation) > 0:
        warnings.append(
            "Any negative non-translation mode beyond the configured "
            "tolerance is treated as a structural-stability failure."
        )
    elif len(small_negative_nontranslation) > 0:
        warnings.append(
            "All negative non-translation modes are within the configured "
            "numerical tolerance and require review."
        )
    if two_dim and len(negative_flexural) > 0:
        warnings.append(
            "2D polarization diagnostic: "
            f"{len(negative_flexural)} negative mode(s) are ZA-like "
            f"({len(unstable_flexural)} significant, "
            f"{len(small_negative_flexural)} small)."
        )
    if two_dim and len(negative_other) > 0:
        warnings.append(
            "2D polarization diagnostic: "
            f"{len(negative_other)} negative mode(s) are non-ZA "
            f"({len(unstable_other)} significant, "
            f"{len(small_negative_other)} small)."
        )

    cutoff_excluded_nontranslation = np.flatnonzero(
        cutoff_inactive & nontranslation_mask
    )
    cutoff_excluded_positive_nontranslation = np.flatnonzero(
        positive_physical_below_cutoff
    )
    cutoff_excluded_za = np.array([], dtype=int)
    if two_dim:
        cutoff_excluded_za = cutoff_excluded_positive_nontranslation[
            z_polarization[cutoff_excluded_positive_nontranslation]
            >= flexural_min
        ]
    overall_status = gate2_status
    if gate1_status in {"REVIEW", "REVIEW_2D_FLEXURAL"}:
        overall_status = "PASS_WITH_WARNING"
    flexural_fail_allowed = (
        gate1_status == "FAIL_2D_FLEXURAL"
        and allow_2d_flexural_fail
    )
    if flexural_fail_allowed:
        overall_status = "TEST_ONLY_FAIL_2D_FLEXURAL"
        warnings.append(
            "allow_2d_flexural_fail=True: continuing for an explicitly "
            "marked test calculation while excluding negative modes."
        )
    if gate1_status == "FAIL":
        overall_status = "FAIL"
    elif gate1_status == "FAIL_2D_FLEXURAL" and not flexural_fail_allowed:
        overall_status = "FAIL_2D_FLEXURAL"

    diagnostics = {
        "status": overall_status,
        "gate1_status": gate1_status,
        "gate2_status": gate2_status,
        "translation_mode_indices": translation_indices.tolist(),
        "lowest_mode_indices": lowest_indices.tolist(),
        "translation_mode_overlaps": overlaps[translation_indices].tolist(),
        "translation_capture": translation_capture,
        "minimum_translation_overlap": minimum_translation_overlap,
        "asr_residual_ratio": asr_residual_ratio,
        "active_translation_leakage": active_translation_leakage,
        "inactive_translation_capture": inactive_translation_capture,
        "negative_nontranslation_indices": (
            negative_nontranslation.tolist()
        ),
        "negative_nontranslation_frequencies_cm": (
            frequencies_cm[negative_nontranslation].tolist()
        ),
        "negative_nontranslation_z_polarization": (
            z_polarization[negative_nontranslation].tolist()
        ),
        "significant_negative_nontranslation_count": int(
            len(unstable_nontranslation)
        ),
        "significant_negative_nontranslation_indices": (
            unstable_nontranslation.tolist()
        ),
        "significant_negative_nontranslation_frequencies_cm": (
            frequencies_cm[unstable_nontranslation].tolist()
        ),
        "small_negative_nontranslation_count": int(
            len(small_negative_nontranslation)
        ),
        "small_negative_nontranslation_indices": (
            small_negative_nontranslation.tolist()
        ),
        "small_negative_nontranslation_frequencies_cm": (
            frequencies_cm[small_negative_nontranslation].tolist()
        ),
        "negative_za_like_count": int(len(negative_flexural)),
        "negative_za_like_indices": negative_flexural.tolist(),
        "negative_za_like_frequencies_cm": (
            frequencies_cm[negative_flexural].tolist()
        ),
        "negative_za_like_z_polarization": (
            z_polarization[negative_flexural].tolist()
        ),
        "negative_non_za_count": int(len(negative_other)) if two_dim else 0,
        "negative_non_za_indices": (
            negative_other.tolist() if two_dim else []
        ),
        "negative_non_za_frequencies_cm": (
            frequencies_cm[negative_other].tolist() if two_dim else []
        ),
        "negative_non_za_z_polarization": (
            z_polarization[negative_other].tolist() if two_dim else []
        ),
        "cutoff_excluded_mode_count": int(np.sum(cutoff_inactive)),
        "cutoff_excluded_nontranslation_count": int(
            len(cutoff_excluded_nontranslation)
        ),
        "cutoff_excluded_positive_nontranslation_count": int(
            len(cutoff_excluded_positive_nontranslation)
        ),
        "cutoff_excluded_positive_nontranslation_indices": (
            cutoff_excluded_positive_nontranslation.tolist()
        ),
        "cutoff_excluded_positive_nontranslation_frequencies_cm": (
            frequencies_cm[
                cutoff_excluded_positive_nontranslation
            ].tolist()
        ),
        "cutoff_excluded_za_mode_count": int(len(cutoff_excluded_za)),
        "cutoff_excluded_za_mode_indices": cutoff_excluded_za.tolist(),
        "cutoff_excluded_za_frequencies_cm": (
            frequencies_cm[cutoff_excluded_za].tolist()
        ),
        "flexural_polarization_min": float(flexural_min),
        "allow_2d_flexural_fail": allow_2d_flexural_fail,
        "two_dim": two_dim,
        "omega_threshould_cm": float(setup.omega_threshould),
        "warnings": warnings,
    }

    print(
        "mode gates: "
        f"Gate1={gate1_status}, Gate2={gate2_status}, "
        f"status={overall_status}"
    )
    print(
        "translation modes: "
        f"indices={translation_indices.tolist()}, "
        f"capture={translation_capture:.12f}, "
        f"active leakage={active_translation_leakage:.3e}, "
        f"ASR residual={asr_residual_ratio:.3e}"
    )
    for warning in warnings:
        print(f"WARNING: {warning}")

    if gate1_status == "FAIL":
        raise ValueError(
            "Mode Gate 1 failed because significant negative "
            "non-translation modes were found."
        )
    if gate1_status == "FAIL_2D_FLEXURAL" and not flexural_fail_allowed:
        raise ValueError(
            "Mode Gate 1 failed with FAIL_2D_FLEXURAL because significant "
            "negative ZA-like non-translation modes were found."
        )
    if gate2_status == "REVIEW":
        raise ValueError(
            "Mode Gate 2 requires review and omega_threshould does not "
            "isolate the translation subspace."
        )

    active_mask = cutoff_active & ~translation_mask
    return diagnostics, translation_mask, active_mask


def _mean_positive_spacing(frequencies, excluded_mask):
    """Return mean spacing after removing rigid translations."""
    frequencies = np.asarray(frequencies)
    positive = np.sort(frequencies[(frequencies > 0.0) & ~excluded_mask])
    if len(positive) < 2:
        raise ValueError("at least two positive non-translation modes are required")
    return float(np.mean(np.diff(positive)))


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

    #from pyAF.nearest import find_nearest_ortho
    dist=np.zeros((natom,natom,3))
    positions=atoms.positions
    cell=atoms.cell
    for i in range(natom):  
        for j in range(i):
            dist[i,j]=find_nearest_optimized(atoms,i,j)
            #dist[i,j]=find_nearest_ortho(positions,cell,i,j)
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


#from joblib import Parallel, delayed

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
def get_Sij(
    Vx,
    Vy,
    Vz,
    eigenvector,
    omega,
    omega_threshould,
    fix_diag,
    excluded_modes=None,
):

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


    if excluded_modes is None:
        excluded_modes = np.zeros(nmodes, dtype=bool)
    inv_omega=np.zeros(nmodes)
    for i in range(nmodes):
        #tentative
        if omega[i] > omega_threshould and not excluded_modes[i]:
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
    for i in range(nmodes):
        if eigenvalue[i] <0.0:
            val=-np.sqrt(-eigenvalue[i])*pc.scale_cm
            omega.append(val)
        else:
            val=np.sqrt(eigenvalue[i])*pc.scale_cm
            omega.append(val)
    omega=np.asarray(omega)
    mode_gate, translation_mask, active_mask = _evaluate_mode_gates(
        setup,
        eigenvalue,
        eigenvector,
        masses,
        omega,
    )
    Sx,Sy,Sz=get_Sij(
        Vx,
        Vy,
        Vz,
        eigenvector,
        omega,
        setup.omega_threshould,
        setup.fix_diag,
        excluded_modes=translation_mask,
    )

    constant = ((1.0e-17*pc.eV_J*pc.AVOGADRO)**0.5)*(pc.scale_cm**3)
    constant = np.pi*constant/48.0

    if setup.using_mean_spacing:
        dwavg=_mean_positive_spacing(omega, translation_mask)
        print('average mode spacing:{0:8f} cm-1'.format(dwavg))
        broad=setup.broadening_factor*dwavg
    else:
        broad=setup.broadening_factor
    
    Di=np.zeros(len(omega))
    active_indices=np.flatnonzero(active_mask)
    for i in active_indices:
        Di_loc = 0.0
        for j in active_indices:
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
            cv_i = _mode_heat_capacity(xfreq, pc.BOLTZMANN_CONSTANT)
            kappa_info[i]=[omega[i],Di[i]*1.0e4,cv_i*kappafct*Di[i]]
            kf.write('{0:8f}  {1:12f}  {2:12f}\n'.format(omega[i],Di[i]*1.0e4,cv_i*kappafct*Di[i]))

    return {
        'freq':kappa_info[:,0],
        'diffusivity':kappa_info[:,1],
        'thermal_conductivity':kappa_info[:,2],
        'mode_gate':mode_gate,
    }

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
    for i in range(nmodes):
        if eigenvalue[i] <0.0:
            val=-np.sqrt(-eigenvalue[i])*pc.scale_cm
            omega.append(val)
        else:
            val=np.sqrt(eigenvalue[i])*pc.scale_cm
            omega.append(val)
    omega=np.asarray(omega)
    mode_gate, translation_mask, active_mask = _evaluate_mode_gates(
        setup,
        eigenvalue,
        eigenvector,
        masses,
        omega,
    )
    Sx,Sy,Sz=get_Sij(
        Vx,
        Vy,
        Vz,
        eigenvector,
        omega,
        setup.omega_threshould,
        setup.fix_diag,
        excluded_modes=translation_mask,
    )

    constant = ((1.0e-17*pc.eV_J*pc.AVOGADRO)**0.5)*(pc.scale_cm**3)
    #not averaged out for x-,y-,z- dimension
    constant = np.pi*constant/16.0

    if setup.using_mean_spacing:
        dwavg=_mean_positive_spacing(omega, translation_mask)
        print('average mode spacing:{0:8f} cm-1'.format(dwavg))
        broad=setup.broadening_factor*dwavg
    else:
        broad=setup.broadening_factor
    
    #x-,y-,z-direction
    Di=np.zeros((len(omega),3))
    active_indices=np.flatnonzero(active_mask)
    for i in active_indices:
        Di_loc_x = 0.0
        Di_loc_y = 0.0
        Di_loc_z = 0.0
        for j in active_indices:
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
            cv_i = _mode_heat_capacity(xfreq, pc.BOLTZMANN_CONSTANT)
            diffusivity[i]=Di[i]*1.0e4
            kappa[i]=cv_i*kappafct*Di[i]
            kf.write('{0:8f}  {1:8f}  {2:8f} {3:8f} {4:8f} {5:8f} {6:8f}　\n'.
            format(omega[i],diffusivity[i,0],diffusivity[i,1],diffusivity[i,2],
            kappa[i,0],kappa[i,1],kappa[i,2]))

    result = {
        'freq':omega,
        'diffusivity':diffusivity,
        'thermal_conductivity':kappa,
        'mode_gate':mode_gate,
    }
    if setup.two_dim:
        kappa_total = np.sum(kappa, axis=0)
        kappa_summary = {
            'x':float(kappa_total[0]),
            'y':float(kappa_total[1]),
            'z_diagnostic':float(kappa_total[2]),
            'in_plane_average':float(
                0.5 * (kappa_total[0] + kappa_total[1])
            ),
            'unit':'W/mK',
        }
        result['thermal_conductivity_summary'] = kappa_summary
        print(
            "2D thermal conductivity: "
            f"kappa_x={kappa_summary['x']:.12f}, "
            f"kappa_y={kappa_summary['y']:.12f}, "
            "kappa_in_plane="
            f"{kappa_summary['in_plane_average']:.12f} W/mK"
        )
    return result


'''
input is class setup object
using THz unit as frequency unit
(to compare with Shiga-san's code)
'''
def get_thermal_conductivity_THz_unit(setup):
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
    #omega is angular frequency
    for i in range(nmodes):
        if eigenvalue[i] <0.0:
            val=-np.sqrt(-eigenvalue[i])*pc.scale_THz*2.0*np.pi
            omega.append(val)
        else:
            val=np.sqrt(eigenvalue[i])*pc.scale_THz*2.0*np.pi
            omega.append(val)
    omega=np.asarray(omega)
    frequencies_cm = (
        np.sign(eigenvalue)
        * np.sqrt(np.abs(eigenvalue))
        * pc.scale_cm
    )
    mode_gate, translation_mask, active_mask = _evaluate_mode_gates(
        setup,
        eigenvalue,
        eigenvector,
        masses,
        frequencies_cm,
    )
    unit_factor = _angular_thz_per_wavenumber(pc)
    omega_threshold = setup.omega_threshould * unit_factor
    lorentzian_threshold = setup.broadening_threshould / unit_factor
    Sx,Sy,Sz=get_Sij(
        Vx,
        Vy,
        Vz,
        eigenvector,
        omega,
        omega_threshold,
        setup.fix_diag,
        excluded_modes=translation_mask,
    )

    frequency_scale = pc.scale_THz * 2.0 * np.pi
    constant = (
        np.pi
        / 48.0
        * (1.0e-17 * pc.eV_J * pc.AVOGADRO) ** 0.5
        * frequency_scale**3
    )

    if setup.using_mean_spacing:
        dwavg=_mean_positive_spacing(omega, translation_mask)
        print('average mode spacing:{0:8f} 2piTHz'.format(dwavg))
        broad=setup.broadening_factor*dwavg
    else:
        broad=setup.broadening_factor*unit_factor

    #vol: Angstrom^3
    vol = atoms.get_volume()

    Di=np.zeros(len(omega))
    active_indices=np.flatnonzero(active_mask)
    for i in active_indices:
        Di_loc = 0.0
        for j in active_indices:
            dwij = (1.0/np.pi)*broad/( (omega[j] - omega[i])**2 + broad**2 )
            if(dwij > lorentzian_threshold):
                Di_loc = Di_loc + dwij*Sx[j,i]**2+dwij*Sy[j,i]**2+dwij*Sz[j,i]**2
        Di[i] = Di[i] + Di_loc*constant/(omega[i]**2)

    #Di=Di*1.0e-4
    kappafct = 1.0e30/vol
    #1.0e12:Hz to THz
    freqfact = pc.hbar/(pc.BOLTZMANN_CONSTANT*setup.temperature)*1.0e12
    kappa_info=np.zeros((nmodes,3))

    with open('kappa_out_THz'+setup.style,'w') as kf:
        kf.write('frequency[THz]   Diffusivity[cm^2/s]   Thermal_conductivity[W/mK] \n')
        for i in range(nmodes):
            xfreq = omega[i]*freqfact
            cv_i = _mode_heat_capacity(xfreq, pc.BOLTZMANN_CONSTANT)
            kappa_info[i]=[omega[i]/2.0/np.pi,Di[i]*1.0e4,cv_i*kappafct*Di[i]]
            kf.write('{0:8f}  {1:12f}  {2:12f}\n'.format(omega[i]/2.0/np.pi,Di[i]*1.0e4,cv_i*kappafct*Di[i]))

    return {
        'freq':kappa_info[:,0],
        'diffusivity':kappa_info[:,1],
        'thermal_conductivity':kappa_info[:,2],
        'mode_gate':mode_gate,
    }
