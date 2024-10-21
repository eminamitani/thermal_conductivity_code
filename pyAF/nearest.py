
import numpy as np
#simple algorith for ortholombic case
def find_nearest_ortho(positions,cell,i,j):
    import numpy as np
    distance=positions[j]-positions[i]
    rv=cell
    #cell is ortholombic, so only diagonal element should be considered
    xinit=distance[0]-2.0*rv[0,0]
    yinit=distance[1]-2.0*rv[1,1]
    zinit=distance[2]-2.0*rv[2,2]

    #consider distance between equiliblium 27=3x3x3 cell
    ii=np.array([i//9+1 for i in range(27)],dtype=float)
    jj=np.array([(i//3)%3+1 for i in range(27)],dtype=float)
    kk=np.array([i%3+1 for i in range(27)],dtype=float)

    xcan=xinit+rv[0,0]*ii
    ycan=yinit+rv[1,1]*jj
    zcan=zinit+rv[2,2]*kk

    candidate=np.stack((xcan,ycan,zcan),axis=1)
    dist=[np.linalg.norm(candidate[i,:]) for i in range(27)]
    min=np.min(dist)
    index=np.max(np.where(dist==min))

    return candidate[index]


# MIC distance evaluation in ASE atoms object
def find_nearest_optimized(atoms, i, j):
    # direct distances
    distance = atoms.get_distance(i, j, vector=True)
    
    # cell vectors
    rv = atoms.cell
    
    # 27 candidates
    shifts = np.array([[-1, 0, 1]]).T
    ii, jj, kk = np.meshgrid(shifts, shifts, shifts)
    
    # all shifts
    candidates = distance + ii.flatten()[:, None] * rv[0] + \
                             jj.flatten()[:, None] * rv[1] + \
                             kk.flatten()[:, None] * rv[2]
    
    # calculate distances
    dists = np.linalg.norm(candidates, axis=1)
    
    # minimal
    min_index = np.argmin(dists)
    return candidates[min_index]



