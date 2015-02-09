# -*- coding: utf8 -*-

import numpy
from matplotlib import pyplot
import random
import math

def load_data():
  global Nt, Np, N, t, p, matrix
  t = numpy.loadtxt('data/taillepoids_f.txt')
  Nt = len(t)
  p = numpy.loadtxt('data/taillepoids_h.txt')
  Np = len(p)
  matrix = numpy.concatenate((t, p), axis=0)
  N = Nt + Np

def print_graphs():
  pyplot.figure(1)
  pyplot.plot(t[:,0],t[:,1], '.')
  pyplot.plot(p[:,0],p[:,1], '.')

  #pyplot.figure(2)
  #pyplot.plot(p, '.')

  pyplot.show()
  
def pas_batch(val):
  A = 100.0
  B = 1
  C = 10000
  return ((A/(C + (B * val))))
  
def j_theta(theta):
  tmp = (matrix[1,:] - numpy.dot(matrix[0,:].T, theta))
  # numpy.dot(tmp.T, tmp) = le carré
  return ((1.0/N) * numpy.dot(tmp.T, tmp))
  
def batch_gradient_descent():
  theta = [1, 1]

  previous = j_theta(theta)
  current = previous + 1

  i = 1
  while (previous != current):
    previous = current
    theta = theta + (pas_batch(i) * (1.0/N) * numpy.dot(matrix[0,:], (matrix[1,:] - (1.0/(1 + (math.exp(numpy.dot(theta.T, matrix[0,:]))))))))
    i+=1
    current = j_theta(theta)

  print "Batch theta = ", theta
  return theta

def main():
  load_data()
  batch_gradient_descent()
  print_graphs()

if __name__ == '__main__':
  main()
  
