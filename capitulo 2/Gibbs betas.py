# muestreo de Gibbs Betas
import numpy as np
from matplotlib import pyplot as plt
import time
import scipy as sp
from scipy import stats
import math
from decimal import *
import mpmath as mp
from statsmodels.graphics.tsaplots import plot_acf

tim1=time.clock()

n_a=40 # número de observaciones provenientes de theta A
n_b=40 # número de observaciones provenientes de theta B
np.random.seed(345)
theta=np.array([0.7,0.66])
x=np.random.binomial(1,theta[0],size=n_a) # observaciones provenientes de theta A
y=np.random.binomial(1,theta[1],size=n_b) # observaciones provenientes de theta B
s_a=np.sum(x)
s_b=np.sum(y)



def fact(num): # función para generar factoriales
  prod = Decimal(1.0)
  cero = 0.0
  num_aux = np.array([num]).astype(np.int)
  num_aux=num_aux[0]
  for ii in range(num_aux):
    aux = Decimal(ii+1)
    prod = prod*aux
  if num==0.0:
    prod=Decimal(1.0)
  return prod

def f(theta_eval): # función de probabilidad conjunta sin normalizar
  res = s_a*np.log(theta_eval[0])+(n_a-s_a)*np.log(1-theta_eval[0])+s_b*np.log(theta_eval[1])+(n_b-s_b)*np.log(1-theta_eval[1])
  return np.exp(res)

def sim_kernel(b,w): #función que simula observaciones de la cadena
  if w==1:
    alf = s_a + 1
    bet = n_a - s_a + 1
    aux=np.random.beta(alf,bet)
    res = b.copy()
    res[0]=aux
  elif w==2:
    alf = s_b + 1
    bet = n_b - s_b + 1
    aux=np.random.beta(alf,bet)
    res = b.copy()
    res[1]=aux
  return res

def kernel(vec1,w): # función que evalúa el kernel de la propuesta
  if w==1:
    alf = s_a + 1
    bet=n_a-s_a+1
    res_aux=sp.stats.beta(alf,bet)
    res = res_aux.pdf(vec1[0])
  elif w==2:
    alf = s_b + 1
    bet=n_b-s_b+1
    res_aux=sp.stats.beta(alf,bet)
    res = res_aux.pdf(vec1[1])
  return res

def rho(vec1,vec2,w): # función que evalúa la probabilidad de aceptación
  if vec2[0]<vec2[1]:
    div=0.0
  else:
    div_aux = np.array([f(vec2)/f(vec1)]).astype(np.float64)[0]
    div = div_aux*kernel(vec1,w)/kernel(vec2,w)
  res = np.min(np.array([1.0,div]))
  return res


inicial = np.array([0.1,0.05]) #punto inicial
cadena = np.column_stack((np.zeros(2),inicial))
lim = 100 # tamaño de muestra
cont = 0
probs = np.array([0.5,0.5]) #probabilidad para cada propuesta
logver = np.array([])
lag = 10
muestra = np.zeros((2,1)) # arreglo para guardar simulaciones
muest = 0 #contador de muestras que se han simulado
rech = 0 # contador de rechazos
burn_in = 10

while muest<lim:
  indicador = np.random.choice(range(1,3),p=probs) # se elige distribución propuesta
  sh=np.shape(cadena) 
  prop = sim_kernel(cadena[:,sh[1]-1],indicador) # se obtiene propuesta
  aux = rho(cadena[:,sh[1]-1],prop,indicador)
  u = np.random.random()
  if u<=aux:
    logver = np.concatenate((logver,np.array([np.log(f(prop))]))) # se obtiene logverosimilitud
    cadena = np.column_stack((cadena,prop)) # se incorpora propuesta a cadena
  else:
    logver = np.concatenate((logver,np.array([np.log(f(cadena[:,sh[1]-1]))])))
    rech = rech + 1
    cadena = np.column_stack((cadena,cadena[:,sh[1]-1]))
  if cont%lag==0 and cont>0 and burn_in <cont:
    muestra = np.column_stack((muestra, cadena[:,sh[1]-1])) # se integra observación en muestra pseudoindependiente
    muest = muest + 1
  cont = cont +1

tim2=time.clock()

timf=tim2-tim1
timf_m=timf/60.0

def uniforme1(a,b):
  if b>0 and b <=a and a<1:
    ff=s_a*np.log(a)+(n_a-s_a)*np.log(1-a)+s_b*np.log(b)+(n_b-s_b)*np.log(1-b)
    ff=np.exp(ff)
  else:
    ff=0.0
  return ff

parta=125
partb=125
Z=np.zeros((parta,partb))

ainf=0.0
asup=1.0
binf=0.0
bsup=1.0

a=0
for i in np.linspace(ainf,asup,parta):
  b=0
  for j in np.linspace(binf,bsup,partb):
      Z[a,b]=uniforme1(i,j)
      b=b+1
  a=a+1

menos = 50
plt.contourf(np.linspace(ainf,asup,parta),np.linspace(binf,bsup,partb),np.transpose(Z),levels=50)
plt.colorbar().ax.set_ylabel('densidad no normalizada')
plt.plot(cadena[0,1:menos+1],cadena[1,1:menos+1],color='orange')
plt.show()

menos = 300
tiempo = range(1,menos+1)
plt.plot(tiempo,logver[0:menos])
plt.axvline(x=burn_in,color='r')
plt.xlabel('t')
plt.ylabel('ln(q)')
plt.show()


fig, axs = plt.subplots(2)
fig = plot_acf(cadena[0,burn_in:],ax=axs[0],marker="",lags=15,title='')
fig = plot_acf(cadena[1,burn_in:],ax=axs[1],marker="",lags=15,title='')
fig.text(0.5, 0.04, 'lag', ha='center')
fig.text(0.04, 0.5, 'Autocorrelación', va='center', rotation='vertical')
plt.show()


media_sim = np.mean(muestra,axis=1)




l1=np.linspace(0,1,num=10)
plt.contourf(np.linspace(ainf,asup,parta),np.linspace(binf,bsup,partb),np.transpose(Z),levels=50)
plt.colorbar().ax.set_ylabel('densidad no normalizada')
#plt.scatter(muestra[0,1:lim+1],muestra[1,1:lim+1],facecolors="none",edgecolors="xkcd:lavender")
#plt.scatter(media_sim[0],media_sim[1],facecolors="r")
#plt.plot(l1,l1,color="r")
plt.show()

l1=np.linspace(0,1,num=10)
plt.contourf(np.linspace(ainf,asup,parta),np.linspace(binf,bsup,partb),np.transpose(Z),levels=50)
plt.colorbar().ax.set_ylabel('densidad no normalizada')
plt.scatter(muestra[0,1:lim+1],muestra[1,1:lim+1],marker='o',color="orange",facecolors='none',label='Muestra',linewidth=1.5)
plt.legend()
#plt.scatter(media_sim[0],media_sim[1],facecolors="r")
#plt.plot(l1,l1,color="r")
plt.show()



print("Segundos: ", timf,sep="")
print("Minutos: ", timf_m,sep="")
print("")
print("Iteraciones: ", cont,sep="")
print("")
print('Estimador puntual theta A:',np.round(media_sim[0],4),sep="")
print('Estimador puntual theta B:',np.round(media_sim[1],4),sep="")
