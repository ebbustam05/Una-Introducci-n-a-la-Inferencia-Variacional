# Metropolis-Hastings Weibull
import numpy as np
from matplotlib import pyplot as plt
import time
import scipy as sp
from scipy import stats
import math
from decimal import *
import mpmath as mp
from statsmodels.graphics.tsaplots import plot_acf



np.random.seed(345)

#parámetros weibull
c=1.0
k=1.5

#parámetros normal
mu=0
sigma=0.5

def f(weib): # función de densidad Weibull
  res = np.log(c)+np.log(k)+(k-1)*np.log(weib)-c*weib**k
  return np.exp(res)

def sim_kernel(b,w): #función que simula observaciones de la cadena
  if w==1:
    aux= np.random.normal(loc=mu,scale=sigma)
    res = b + aux
  elif w==2:
    res = b
  return res

def kernel(num1,num2,w): # función que evalúa en la densidad propuesta
  if w==1:
    res_aux=sp.stats.norm(loc=mu,scale=sigma)
    res = res_aux.pdf(num1-num2)
  else:
    res = 1.0
  return res

def rho(num1,num2,w): # función que evalúa la probabilidad de aceptación
  if num2<=0:
    div=0.0
  else:
    div_aux = np.array([f(num2)/f(num1)]).astype(np.float64)[0]
    div = div_aux*kernel(num1,num2,w)/kernel(num2,num1,w)
  res = np.min(np.array([1.0,div]))
  return res

inicial = np.random.random(1)*2+10 # inicio aleatorio de la cadena
cadena = np.concatenate((np.zeros(1),inicial)) 
lim = 1000 # tamaño de muestra
cont = 0
probs = np.array([1.0,0.0]) #probabilidad para cada propuesta
logver = np.array([])
lag = 30
muestra = np.zeros(1) # arreglo para guardar simulaciones
muest = 0 #contador de muestras que se han simulado
rech = 0 # contador de rechazos
burn_in = 65

while muest<lim:
  indicador = np.random.choice(range(1,3),p=probs) # se elige distribución propuesta
  sh=np.shape(cadena) 
  prop = sim_kernel(cadena[sh[0]-1],indicador) # se obtiene propuesta
  aux = rho(cadena[sh[0]-1],prop,indicador)
  u = np.random.random()
  if u<=aux:
    logver = np.concatenate((logver,np.array([np.log(f(prop))]))) # se obtiene logverosimilitud
    cadena = np.concatenate((cadena,np.array([prop]))) # se incorpora propuesta a cadena
  else:
    logver = np.concatenate((logver,np.array([np.log(f(cadena[sh[0]-1]))]))) # se obtiene logverosimilitud
    rech = rech + 1
    cadena = np.concatenate((cadena,np.array([cadena[sh[0]-1]]))) # se repite última observación de la cadena
  if cont%lag==0 and cont>0 and burn_in <cont:
    muestra = np.concatenate((muestra, np.array([cadena[sh[0]-1]]))) # se integra observación en muestra pseudoindependiente
    muest = muest + 1
  cont = cont +1


def posterior(num): # densidad posterior
  if num>0:
    ff=np.log(c)+np.log(k)+(k-1)*np.log(num)-c*num**k
    ff=np.exp(ff)
  else:
    ff=0.0
  return ff

part=125
Z=np.zeros(part)

inf=0.0
sup=5.0

a=0
for i in np.linspace(inf,sup,part):
  Z[a]=posterior(i)
  a=a+1


menos = len(logver)
tiempo = range(1,menos+1)
plt.plot(tiempo,cadena[1:menos+1])
plt.xlabel('n')
plt.ylabel('X_n')
plt.show()

menos = 300
tiempo = range(1,menos+1)
plt.plot(tiempo,cadena[1:menos+1])
plt.xlabel('n')
plt.ylabel('X_n')
plt.show()

l1=np.linspace(inf,sup,num=len(Z))
plt.hist(cadena,density=True,bins=100,label='Simulaciones cadena completa')
plt.plot(l1,Z,label='Densidad real',linewidth=3)
plt.legend()
plt.show()

menos = 300
tiempo = range(1,menos+1)
plt.plot(tiempo,logver[0:menos])
plt.axvline(x=burn_in,color='r')
plt.xlabel('n')
plt.ylabel('ln(q)')
plt.show()


fig = plot_acf(cadena[burn_in:],marker="",lags=70)
plt.xlabel('lag')
plt.ylabel('Autocorrelación')
plt.show()


a=0
for i in np.linspace(inf,sup,part):
  Z[a]=posterior(i)
  a=a+1

media_sim = np.mean(muestra)



l1=np.linspace(inf,sup,num=len(Z))
plt.hist(muestra,density=True,bins=50,label='Simulaciones')
plt.plot(l1,Z,label='Densidad real',linewidth=3)
plt.legend()
plt.show()
