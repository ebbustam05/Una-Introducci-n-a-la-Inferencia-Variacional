# AVB ejemplo betas

import os
import pystan
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from tensorflow.python.framework import ops
#pip install ite
import ite
from tqdm import tqdm_notebook
import re
import glob
import tensorflow as tf
import tensorflow_probability as tfp
import time
import mpmath as mp


np.random.seed(145)
tf.random.set_seed(289612)


t_ini=time.time()

n_a=40 # número de observaciones provenientes de theta A
n_b=40 # número de observaciones provenientes de theta B
theta = np.array([0.7,0.66])
x_a = np.random.binomial(1,theta[0],size=n_a) # observaciones provenientes de theta A
x_b = np.random.binomial(1,theta[1],size=n_b) # observaciones provenientes de theta B

s_a = np.sum(x_a)
s_b = np.sum(x_b)
tam=100 # tamaño muestra aproximante

# Parameters
batch_size = 512
data = {'J': 2, # dimensión de distribución posterior
        'n_a': n_a,
        'n_b': n_b,
        's_a': s_a,
        's_b': s_b,}



param_dim = 2 # dimensión de parámetro

def get_logprob(z, data):
    x1 = z[:, 0:1]
    x2 = z[:, 1:2]

    n_a = tf.constant(data['n_a'], dtype=tf.float32, shape=(1, 1))
    n_b = tf.constant(data['n_b'], dtype=tf.float32, shape=(1, 1))
    s_a = tf.constant(data['s_a'], dtype=tf.float32, shape=(1, 1))
    s_b = tf.constant(data['s_b'], dtype=tf.float32, shape=(1, 1))

    logprob = tf.reduce_sum(
        s_a*tf.math.log(x1) +(n_a-s_a)*tf.math.log(1-x1) + s_b*tf.math.log(x2) +(n_b-s_b)*tf.math.log(1-x2) , [1] ) # logaritmo de kernel verosimilitud
    
    return logprob



red_posterior =  tf.keras.Sequential(
        [
         tf.keras.layers.Dense(128, activation=tf.nn.elu),
         tf.keras.layers.Dense(128, activation=tf.nn.elu),
         tf.keras.layers.Dense(param_dim, activation=tf.nn.sigmoid),
        ]
    )

def posterior(num):
        eps = tfp.distributions.MultivariateNormalDiag(loc=tf.zeros([num, param_dim]), scale_diag=tf.ones([num, param_dim])).sample()
        z=red_posterior(eps)
        return z

red_adversaria =  tf.keras.Sequential(
        [
         tf.keras.layers.Dense(256, activation=tf.nn.elu),
         tf.keras.layers.Dense(256, activation=tf.nn.elu),
         tf.keras.layers.LeakyReLU(alpha=0.2),
         tf.keras.layers.Dense(1,activation=None,kernel_initializer=tf.keras.initializers.zeros()),
        ]
    )

def adversary(z):
  T = red_adversaria(z)
  T = tf.squeeze(T, [1])
  return T

def compute_loss():
  z0_a = tf.random.uniform(shape=(batch_size,1),dtype=tf.dtypes.float32)
  z0_b = tf.random.uniform(shape=(batch_size,1),dtype=tf.dtypes.float32, minval=0., maxval=z0_a)
  z0 = tf.concat([z0_a,z0_b],axis=1)
  z_ = posterior(batch_size)
  beta = tf.constant(1.)

  Ti = adversary(z0)
  Td = adversary(z_)


  logprob = get_logprob(z_, data)
  mean_logprob = tf.reduce_mean(logprob)
  mean_Td = tf.reduce_mean(Td)
  loss_primal = tf.reduce_mean(beta*(Td) - logprob)

  d_loss_d = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=Td, labels=tf.ones_like(Td))
  )
  d_loss_i = tf.reduce_mean(
      tf.nn.sigmoid_cross_entropy_with_logits(logits=Ti, labels=tf.zeros_like(Ti))
  )
  loss_dual = d_loss_i + d_loss_d

  return loss_primal, loss_dual

optimizer_gen = tf.keras.optimizers.Adam(2e-5, beta_1=0.5)
optimizer_disc = tf.keras.optimizers.Adam(1e-3, beta_1=0.5)

@tf.function
def train_step(red_posterior, red_adversaria, optimizer_gen, optimizer_disc):
  """Executes one training step and returns the loss.

  This function computes the loss and gradients, and uses the latter to
  update the model's parameters.
  """
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    loss_aux = compute_loss()
    loss = loss_aux[0]
    disc_loss = loss_aux[1]
  gradients_gen = gen_tape.gradient(loss, red_posterior.trainable_variables)
  gradients_disc = disc_tape.gradient(disc_loss, red_adversaria.trainable_variables)
  optimizer_gen.apply_gradients(zip(gradients_gen, red_posterior.trainable_variables))
  optimizer_disc.apply_gradients(zip(gradients_disc, red_adversaria.trainable_variables))

n_epoch = 5000



vec_gen=np.array([])
vec_disc=np.array([])
for epoch in range(1, n_epoch + 1):
  start_time = time.time()
  train_step(red_posterior, red_adversaria, optimizer_gen, optimizer_disc)
  end_time = time.time()

  aux_loss=compute_loss()
  gen_loss=aux_loss[0]
  disc_loss=aux_loss[1]
  vec_gen=np.concatenate([vec_gen,np.array([gen_loss])])
  vec_disc=np.concatenate([vec_disc,np.array([disc_loss])])
  
t_fin=time.time()

def muestra(red_posterior,num):
  z_ = posterior(num)
  return z_

plt.figure()
plt.plot(np.arange(0,n_epoch), np.asarray(vec_gen[0:]))
plt.xlabel('Epoch')
plt.ylabel('Pérdida red codificadora')
plt.show()

plt.figure()
plt.plot(np.arange(0,n_epoch), np.asarray(vec_disc[0:]))
plt.xlabel('Epoch')
plt.ylabel('Pérdida red discriminativa')
plt.show()

plt.figure()
plt.plot(np.arange(500,n_epoch), np.asarray(vec_gen[500:]))
plt.xlabel('Epoch')
plt.ylabel('Pérdida red codificadora')
plt.show()

plt.figure()
plt.plot(np.arange(500,n_epoch), np.asarray(vec_disc[500:]))
plt.xlabel('Epoch')
plt.ylabel('Pérdida red discriminativa')
plt.show()

t_ini_m=time.time()
n_vis = 1
enc_test = np.vstack([muestra(red_posterior,tam) for _ in range(n_vis)])
t_fin_m=time.time()

l1=np.linspace(0,1,num=10)


def posterior_distr(a,b):
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
      Z[a,b]=posterior_distr(i,j)
      b=b+1
  a=a+1

l1=np.linspace(0,1,num=10)
plt.contourf(np.linspace(ainf,asup,parta),np.linspace(binf,bsup,partb),np.transpose(Z),levels=20)
plt.colorbar().ax.set_ylabel('densidad no normalizada')
plt.scatter(enc_test[0:tam,0], enc_test[0:tam,1],marker='o',color="orange",facecolors='none',label='Muestra',linewidth=1.5)
#plt.scatter(theta[0],theta[1],facecolors="r")
#plt.plot(l1,l1,color="r")
plt.show()

media=np.mean(enc_test,0)

print('Segundos: ', np.round(t_fin-t_ini,decimals=2))
print('Minutos: ', np.round((t_fin-t_ini)/60.,decimals=2))
print("")
print('Segundos muestreo: ', np.round(t_fin_m-t_ini_m,decimals=6))
print("")
print("media: ",media,sep="")
print("")
print("Reales")
print("medias: ",theta,sep="")
