#densidad beta a priori y posterior

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.stats

np.random.seed(935)

theta=0.8 #valor desconocido

n=12 #tamaño de muestra

x=np.random.binomial(1,theta,size=n) #muestra

s=np.sum(x)

#parámetros distribución a priori
alfa_pr=0.8
beta_pr=1.5

#parámetros distribución posterior
alfa_post = alfa_pr + s
beta_post = beta_pr + n - s


media_priori = alfa_pr/(alfa_pr+beta_pr)
estimador_post = alfa_post/(alfa_post+beta_post)

mediana_priori = sp.stats.beta.median(alfa_pr, beta_pr)
mediana_post = sp.stats.beta.median(alfa_post, beta_post)

t = np.linspace(0.001,
                0.999, 100)
plt.plot(t, sp.stats.beta.pdf(t, alfa_pr, beta_pr),
       'r-', label='priori ')
plt.plot(t, sp.stats.beta.pdf(t, alfa_post, beta_post),
       'b-', label='posterior ')
plt.legend()
plt.show()



print('muestra: ', x,sep="")
print('media priori: ', np.round(media_priori,4),sep="")
print('media posterior: ', np.round(estimador_post,4),sep="")
print('mediana priori: ', np.round(mediana_priori,4),sep="")
print('mediana posterior: ', np.round(mediana_post,4),sep="")
