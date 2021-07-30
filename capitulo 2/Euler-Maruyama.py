#Movimiento Browniano Geométrico

#Solución exacta
import numpy as np
from matplotlib import pyplot as plt
import time
import scipy as sp
from scipy import stats
import scipy.special

np.random.seed(38306)

#Tiempo discretizado a considerar
tiempo=np.linspace(0.001,5,num=1000)

#número de trayectorias a simular
tam=20

 # consideramos dX(t)=mu*X(t)*dt+sigma*X(t)dW(t) con estado inicial x0
mu=0.4
sigma=0.1
x0=1.
mu_hat=mu-0.5*sigma**2

#parámetros de la distribución lognormal solución: mu_log, sigma_log
mu_log=mu*tiempo+np.log(x0)-0.5*sigma**2*tiempo
sigma_log=sigma*np.sqrt(tiempo)
w=np.zeros(len(tiempo)-1)
xt=np.zeros(len(tiempo))+x0

# simulación solución exacta
for i in range(tam):
  z=np.random.normal(size=len(tiempo))
  w[0]=0
  for j in range(len(tiempo)-2):
    w[j+1]=w[j]+z[j+1]*np.sqrt(tiempo[1]-tiempo[0])
  xt[1:]=np.exp(np.log(x0)+mu_hat*tiempo[1:]+sigma*w)
  plt.plot(tiempo,xt,'b')
plt.plot(tiempo,sp.stats.lognorm.ppf(0.5,s=sigma_log,loc=0,scale=np.exp(mu_log)),'r--')
plt.plot(tiempo,sp.stats.lognorm.ppf(0.01,s=sigma_log,loc=0,scale=np.exp(mu_log)),'r--')
plt.plot(tiempo,sp.stats.lognorm.ppf(0.99,s=sigma_log,loc=0,scale=np.exp(mu_log)),'r--')
plt.xlabel('t')
plt.ylabel('Xt')
plt.show()



# Euler-Maruyama
incr=tiempo[1]-tiempo[0]
def fun(yi,ti):
  c=sigma*yi
  return c
cadena=np.repeat(x0,(1000))
b=np.repeat(0.0,(1000))
for j in range(tam):
  W=np.random.normal(loc=0.0,scale=np.sqrt(incr),size=1000)
  for i in range(1000-1):
    b[i]=fun(cadena[i],tiempo[i])
    cadena[i+1]=cadena[i]+mu*cadena[i]*(tiempo[i+1]-tiempo[i])+b[i]*W[i]
  plt.plot(tiempo,cadena,'g')
plt.plot(tiempo,sp.stats.lognorm.ppf(0.5,s=sigma_log,loc=0,scale=np.exp(mu_log)),'r--')
plt.plot(tiempo,sp.stats.lognorm.ppf(0.01,s=sigma_log,loc=0,scale=np.exp(mu_log)),'r--')
plt.plot(tiempo,sp.stats.lognorm.ppf(0.99,s=sigma_log,loc=0,scale=np.exp(mu_log)),'r--')
plt.xlabel('t')
plt.ylabel('Xt')
plt.show()
