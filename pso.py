#!/usr/bin/python3.4
# (c) Mohammad H. Mofrad, 2017 
# (e) hasanzadeh@cs.pitt.edu
# Particle Swarm Optimization (PSO)

import numpy as np

def sphere(x):
   size, dim = x.shape
   y = np.array(np.sum(np.power(x,2), axis=1)).reshape(size,1)
   return(y)

def rosenbrock(x):
   size, dim = x.shape
   y = np.sum(100 * np.power(x[:,1:dim] - np.power(x[:,0:dim-1],2), 2) + np.power(x[:,0:dim-1] - 1, 2), axis=1).reshape(size,1)
   return(y)

def ackley(x):
   size, dim = x.shape
   y = np.sum(np.power(x,2), axis=1)
   y = np.array(-20 * np.exp(-0.2 * np.sqrt(y/dim)) - np.exp(np.sum(np.cos(2 * pi * x), axis=1) / dim) + 20 + np.exp(1)).reshape(size,1)
   return(y)

def griewanks(x):
   size, dim = x.shape
   y = np.array(np.sum(np.power(x, 2)/4000, axis=1) - np.prod(np.cos(x/np.sqrt(np.arange(dim) + 1)), axis = 1) + 1).reshape(size,1)
   return(y)

def rastrigin(x):
   size, dim = x.shape
   y = np.array(np.sum(np.power(x, 2) - 10 * np.cos(2 * pi * x), axis = 1) + 10 * dim).reshape(size, 1)
   return(y)

benchmarks = {'sphere':sphere , 'ackley':ackley, 'rosenbrock':rosenbrock, 'griewanks':griewanks, 'rastrigin':rastrigin}

f = 'sphere'
if f == 'sphere':
   xmax =  100
elif f == 'ackley':
   xmax = 32.768
elif f == 'rosenbrock':
   xmax = 2.048
elif f == 'griewanks':
   xmax = 600
elif f == 'rastrigin':
   xmax = 5.12

# Maximum iterations
imax = 1000

# Acceleration coefficients
c1   = 1.49445
c2   = 1.49445

# Weight
wmax = 0.9
wmin = 0.4

# Number of dimensions
dim  = 30

# Population size
size = 50

# Debug level
VERBOS = True

# Max and min position bounds
xmax =  100
xmin = -xmax

# Max and min velocity bounds
vmax = 0.2 * (xmax - xmin)
vmin = -vmax

# Position
x    = np.zeros((size, dim))
x    = xmin + ((xmax - xmin) * np.random.rand(size, dim))

# Velocity
v    = np.zeros((size, dim))
v    = vmin + ((vmax - vmin) * np.random.rand(size, dim))

# Fitness
fx   = np.zeros((size,1))
fx   = benchmarks[f](x)

# Personal best position
pb   = np.zeros((size, dim))
fpb  = np.zeros((size, 1))
pb   = np.copy(x)
fpb  = np.copy(fx)

# Global best position
gb   = np.zeros((1, dim))
fgb  = 0
gb   = np.copy(x[np.argmin(fx), :].reshape(1,dim))
fgb  = np.copy(fx[np.argmin(fx)]).reshape(1,1)

w = 0.74

# Main loop for updating particles
for i in range(imax):
   for k in range(size):
      # Update partcile's velocity
      v[k,:] = (w * v[k,:]) + (c1 * np.random.rand(1, dim) * (pb[k,:] - x[k,:])) + (c2 * np.random.rand(1, dim) * (gb[0,:] - x[k,:]))

      # Update particle's position 
      x[k,:] = x[k,:] + v[k,:]

      # Update particle's fitness
      # Apply TDR sub dimensions
      fx[k,0] = benchmarks[f](x[k,:].reshape(1,dim))
      
      # Update personal best position	
      if(fx[k,0] < fpb[k,0]):
         pb[k,:]  = x[k,:]
         fpb[k,0] = fx[k,0]

      # Update global best position
      if(fx[k,0] < fgb):
         gb[0,:] = x[k,:]
         fgb = fx[k,0]
      if VERBOS:
         print(' Iteration', i, 'Global best', fgb)
