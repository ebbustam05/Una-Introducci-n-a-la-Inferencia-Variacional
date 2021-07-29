#Descenso de Gradiente Estocástico

import numpy as np
from matplotlib import pyplot as plt
import time
import scipy as sp
from scipy import stats
import scipy.special

np.random.seed(81934)

#función a optimizar
def function(x,y):
  f=x**2+y**2+x+y+x*y
  return f

#derivadas parciales
def parcial(x,y):
  res1 = 2*x+y+1
  res2 = 2*y+x+1
  return np.array([res1,res2])

dif=1.
minimo=np.array([-10,-30])
aux=np.zeros(2)
vec=minimo.copy()
contador=1
while dif>0.01:
  aux=minimo.copy()
  minimo=aux-(parcial(aux[0],aux[1])+np.random.normal(loc=0.,scale=30.0,size=2))/(2*contador)
  contador=contador+1
  dif=np.sum(np.abs(minimo-aux))
  vec=np.column_stack((vec,np.reshape(minimo,(2,1))))


parta=125
partb=125
Z=np.zeros((parta,partb))

ainf=-35.0
asup=35.0
binf=-35.0
bsup=35.0

a=0
for i in np.linspace(ainf,asup,parta):
  b=0
  for j in np.linspace(binf,bsup,partb):
      Z[a,b]=function(i,j)
      b=b+1
  a=a+1

plt.contour(np.linspace(ainf,asup,parta),np.linspace(binf,bsup,partb),np.transpose(Z),levels=10,colors="black")
plt.plot(vec[0,:],vec[1,:],'rx')
plt.plot(vec[0,:],vec[1,:],'r',linewidth=0.5,linestyle='dashed')
plt.plot(minimo[0],minimo[1],'bo')
plt.show()

print('Mínimo aproximado: ',minimo,sep='')
