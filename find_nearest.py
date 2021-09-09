from ase.io import read
import ase.neighborlist

atoms=read('data.vasp',format='vasp')

#mic sometimes impose strong constrain
#it is better to compare with/without mic and then determin the minimal distance
r1=atoms.get_distance(0,1,vector=True)
r1_mic=atoms.get_distance(0,1,mic=True,vector=True)
print(r1,r1_mic)

r2=atoms.get_distance(0,2,vector=True)
r2_mic=atoms.get_distance(0,2,mic=True,vector=True)
print(r2,r2_mic)

r3=atoms.get_distance(0,3,vector=True)
r3_mic=atoms.get_distance(0,3,mic=True,vector=True)
print(r3,r3_mic)

r4=atoms.get_distance(0,4,vector=True)
r4_mic=atoms.get_distance(0,4,mic=True,vector=True)
print(r4,r4_mic)

r5=atoms.get_distance(0,5,vector=True)
r5_mic=atoms.get_distance(0,5,mic=True,vector=True)
print(r5,r5_mic)


r6=atoms.get_distance(0,6,vector=True)
r6_mic=atoms.get_distance(0,6,mic=True,vector=True)
print(r6,r6_mic)

r7=atoms.get_distance(0,7,vector=True)
r7_mic=atoms.get_distance(0,7,mic=True,vector=True)
print(r7,r7_mic)