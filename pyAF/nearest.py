

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

#find nearest image and return vector
#atoms: ase.Atoms object
#simply search all image atoms
#just python version of GULP implementention
#there should be more efficient way

def find_nearest(atoms,i,j):
    #for 3-dim case

    distance=atoms.get_distance(i,j,vector=True)
    xdc=distance[0]
    ydc=distance[1]
    zdc=distance[2]

    rmin=1000.0

    rv=atoms.cell

    if (atoms.cell.shape[0] ==3):
        #set
        xcdi=xdc-2.0*rv[0,0]
        ycdi=ydc-2.0*rv[1,0]
        zcdi=zdc-2.0*rv[2,0]

        for ii in [-1,0,1]:
            xcdi=xcdi+rv[0,0]
            ycdi=ycdi+rv[1,0]
            zcdi=zcdi+rv[2,0]

            xcdj=xcdi-2.0*rv[0,1]
            ycdj=ycdi-2.0*rv[1,1]
            zcdj=zcdi-2.0*rv[2,1]

            for jj in [-1,0,1]:
                xcdj=xcdj+rv[0,1]
                ycdj=ycdj+rv[1,1]
                zcdj=zcdj+rv[2,1]
                xcrd = xcdj - 2.0*rv[0,2]
                ycrd = ycdj - 2.0*rv[1,2]
                zcrd = zcdj - 2.0*rv[2,2]

                for kk in [-1,0,1]:
                    xcrd = xcrd + rv[0,2]
                    ycrd = ycrd + rv[1,2]
                    zcrd = zcrd + rv[2,2]
                    r = xcrd*xcrd + ycrd*ycrd + zcrd*zcrd
                    if (r<rmin):
                        rmin = r
                        xdc = xcrd
                        ydc = ycrd
                        zdc = zcrd
    
    return xdc,ydc,zdc,rmin




