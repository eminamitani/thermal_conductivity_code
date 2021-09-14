from ase.io import read

xyz=read('gulp.xyz',format='xyz')
vasp=read('optimized.vasp',format='vasp')
import numpy as np
diff=np.sum(np.abs(xyz.positions-vasp.positions))

print(diff)