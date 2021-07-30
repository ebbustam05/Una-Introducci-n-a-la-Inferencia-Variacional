#CAVI volatilidad
import numpy as np
import math
import time
import scipy as sp
import matplotlib.pyplot as plt
from scipy import special
from scipy import stats

#hiperparámetros
tim1=time.clock()
n=1000
T=1.0
alfa1=0.1
alfaz=20.0
alfa=20.0
m=30
s=57810 #semilla
N=math.floor(n/m)
r=n-m*N
np.random.seed(s) # se fija la semilla

tam_m=1000 # tamaño de la muestra con la que se aproximará

y0=0.0
lamt=T/n
W=np.random.normal(loc=0.0,scale=math.sqrt(lamt),size=n) # generación de normales para incrementos de movimiento browniano

def fun(yi,ti): #función de volatilidad de la SDE
  c=(ti+1)**2
  return c

cadena=np.repeat(y0,(n+1))
b=np.repeat(0.0,(n+1))
y=np.repeat(1.1,n)
for i in range(n):
  b[i]=fun(cadena[i],(i*lamt))
  cadena[i+1]=cadena[i]+b[i]*W[i] # se utiliza Euler-Maruyama para trayectoria de solución de SDE
  y[i]=cadena[i+1]-cadena[i] # se generan variables Y
b[n]=fun(cadena[n],T)

y2=y**2
obs=np.repeat(2.1,N) # vector de observaciones (variable Z)
bi=0
while bi<N:
  suma=0.0
  for i in range(m):
    suma+=y2[m*bi+i]
  obs[bi]=suma
  bi+=1
for i in range(r):
  obs[N-1]+=y2[i+bi*m]

t=np.linspace(0,T,n+1)

dif=0
cont=0
elbo=np.array([0])

# se inicializan parámetros de distribución aproximante
alfa_n=6.98
alfa_1n=0.1
alfa_zn=6.98
vec_alfa=np.repeat(alfa_n,N)
vec_beta=np.repeat(alfa_n,N)
vec_alfaz=np.repeat(alfa_zn,N-1)
vec_betaz=np.repeat(alfa_zn,N-1)

logtheta_n=np.log(vec_beta)-sp.special.digamma(vec_alfa) # E[ln(theta)] para cálculo de ELBO
thetainv_n=vec_alfa/vec_beta # E[(theta)^(-1)]
logzeta_n=np.log(vec_betaz)-sp.special.digamma(vec_alfaz) # E[ln(zeta)] para cálculo de ELBO
zetainv_n=vec_alfaz/vec_betaz # E[(zeta)^(-1)]
while cont<3 or dif>0.000001:
  log_pz=-(m+r)*0.5*logtheta_n[N-1]-0.5*m*np.sum(logtheta_n[0:(N-1)])-0.5/T*n*np.sum(obs*thetainv_n) # auxiliar para ELBO
  suma=0
  for i in range(N-1):
    suma+=zetainv_n[i]*thetainv_n[i+1]
  log_ptzeta=-(alfa1+1)*logtheta_n[0]-alfa1*thetainv_n[0]-(alfa+1)*np.sum(logtheta_n[1:N])-alfa*suma-alfaz*np.sum(logtheta_n[0:(N-1)])-(alfaz+alfa+1)*np.sum(logzeta_n)-alfaz*np.sum(thetainv_n[0:(N-1)]*zetainv_n) # auxiliar para ELBO
  log_qthetaj=np.sum(vec_alfa[1:(N-1)]*np.log(vec_beta[1:(N-1)]))-np.sum(np.log(sp.special.gamma(vec_alfa[1:(N-1)])))-np.sum((vec_alfa[1:(N-1)]+1)*logtheta_n[1:(N-1)])-np.sum(vec_beta[1:(N-1)]*thetainv_n[1:(N-1)]) # auxiliar para ELBO
  log_qtheta1=vec_alfa[0]*np.log(vec_beta[0])-np.log(sp.special.gamma(vec_alfa[0]))-(vec_alfa[0]+1)*logtheta_n[0]-vec_beta[0]*thetainv_n[0] # auxiliar para ELBO
  log_qthetaN=vec_alfa[N-1]*np.log(vec_beta[N-1])-np.log(sp.special.gamma(vec_alfa[N-1]))-(vec_alfa[N-1]+1)*logtheta_n[N-1]-vec_beta[N-1]*thetainv_n[N-1] # auxiliar para ELBO
  log_qzeta=np.sum(vec_alfaz*np.log(vec_betaz))-np.sum(np.log(sp.special.gamma(vec_alfaz)))-np.sum((vec_alfaz+1)*logzeta_n)-np.sum(vec_betaz*zetainv_n) # auxiliar para ELBO
  aux=log_pz+log_ptzeta-log_qthetaj-log_qtheta1-log_qthetaN-log_qzeta # cálculo de ELBO
  elbo=np.concatenate((elbo,np.array([aux]))) # se guarda ELBO
  dif=aux-elbo[cont-1] # se obtiene cambio en ELBO
  cont+=1
  for i in range(N-2):
    vec_alfa[i+1]=0.5*m+alfa+alfaz # se actualizan parámetros de distribución aproximante de theta
    vec_beta[i+1]=0.5*n*obs[i+1]/T+alfa*zetainv_n[i]+alfaz*zetainv_n[i+1] # se actualizan parámetros de distribución aproximante de theta
  vec_alfa[0]=0.5*m+alfa1+alfaz # se actualizan parámetros de distribución aproximante de theta
  vec_beta[0]=n*0.5*obs[0]/T+alfa1+alfaz*zetainv_n[0] # se actualizan parámetros de distribución aproximante de theta
  vec_alfa[N-1]=0.5*(m+r)+alfa # se actualizan parámetros de distribución aproximante de theta
  vec_beta[N-1]=n*0.5*obs[N-1]/T+alfa*zetainv_n[N-2] # se actualizan parámetros de distribución aproximante de theta
  vec_alfaz=np.repeat((alfaz+alfa),(N-1)) # se actualizan parámetros de distribución aproximante de zeta
  for i in range(N-1):
    vec_betaz[i]=alfa*thetainv_n[i+1]+alfaz*thetainv_n[i] # se actualizan parámetros de distribución aproximante de zeta
  logtheta_n=np.log(vec_beta)-sp.special.digamma(vec_alfa)
  thetainv_n=vec_alfa/vec_beta
  logzeta_n=np.log(vec_betaz)-sp.special.digamma(vec_alfaz)
  zetainv_n=vec_alfaz/vec_betaz

tim2=time.clock()
muestra=np.ones(shape=(N,tam_m))
for i in range(N):
  muestra[i,:]=vec_beta[i]*sp.stats.invgamma.rvs(a=vec_alfa[i],size=tam_m) # se obtiene muestra aproximante
muestra=np.sqrt(muestra)
medias=np.mean(muestra,axis=1) # se obtiene media de muestra aproximante
# modas=sp.stats.mode(muestra,axis=1)[0]

t2=np.linspace(0,T,num=(N+1))
esp2=np.repeat(5.4,(N+1))
for i in range(N):
  esp2[i+1]=medias[i]
esp2[0]=medias[0]
tim3=time.clock()
timf1=np.round(tim2-tim1,6)
timf2=np.round(tim3-tim2,6)

err=0.0
ccc=0
for i in range(N):
  for j in range(m):
    err+=(b[ccc+1]-medias[i])**2
    ccc+=1
for i in range(r):
  err+=(b[ccc+1]-medias[N-1])**2
  ccc+=1
act=cont*(2*N-1)

perc=np.ones((N+1,2))*5.3
perc[0,:]=np.percentile(muestra[0,:],np.array([2.5,97.5]))
for i in range(N):
  perc[i+1,:]=np.percentile(muestra[i,:],np.array([2.5,97.5]))

print("Segundos entrenamiento: ", timf1,sep="")
print("Segundos muestra: ", timf2,sep="")
print("Segundos total: ", timf2+timf1,sep="")
print("Iteraciones: ",cont,sep="")
print("Actualizaciones: ",act,sep="")
print("Pérdida cuadrática: ",err,sep="")
print("N=",N,sep="")

#print(np.mean(esp_theta))

plt.plot(range(cont), elbo[1:(cont+1)])
plt.xlabel('Iteración')
plt.ylabel('ELBO')
plt.show()

plt.plot(t,cadena)
plt.xlabel("Tiempo")
plt.ylabel("Xt")
plt.title("Trayectoria")
plt.show()

plt.plot(t,b,label="Real")
plt.step(t2,esp2,label="Estimación")
plt.xlabel("Tiempo")
plt.ylabel("s(t)")
plt.title("Volatilidad")
plt.legend()
plt.show()

plt.plot(t,b,label="Volatilidad",linewidth=2)
plt.step(t2,perc[:,0],linestyle='dashed',color="orange")
plt.step(t2,perc[:,1],linestyle='dashed',color="orange")
plt.xlabel("Tiempo")
plt.ylabel("s(t)")
plt.title("Volatilidad")
plt.legend()
plt.show()

plt.plot(t,b,label="Volatilidad",linewidth=2)
plt.step(t2,perc[:,0],linestyle='dashed',color="orange")
plt.step(t2,perc[:,1],linestyle='dashed',color="orange")
plt.step(t2,esp2,label="Estimación",color='red')
plt.xlabel("Tiempo")
plt.ylabel("s(t)")
plt.title("Volatilidad")
plt.legend()
plt.show()
