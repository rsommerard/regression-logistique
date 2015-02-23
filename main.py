# -*- coding: utf8 -*-

#Batch -> theta

#matrix[:,0] -> tailles
#matrix[:,1] -> poids

import numpy
from matplotlib import pyplot
import random
import math

def load_data():
  global N, matrix, f, h, y, x
  f = numpy.loadtxt('data/taillepoids_f.txt')
  h = numpy.loadtxt('data/taillepoids_h.txt')
  matrix = numpy.concatenate((f, h), axis=0)
  N = len(matrix)
  x = numpy.vstack((matrix[:,0], numpy.ones(N)))
  y = matrix[:,1]

def pas_batch(val):
  A = 1.0
  B = 10000
  C = 1000
  return ((A/(C + (B * val))))

def f_theta(theta):
  return numpy.dot(theta.T, x)

def print_graphs():
  pyplot.figure(1)
  pyplot.plot(f[:,0],f[:,1], '.', label="femme")
  pyplot.plot(h[:,0],h[:,1], '.', label="homme")
  pyplot.plot(matrix[:,0], f_theta(batch_gradient_descent()[-1]), label="theta")
  pyplot.legend()

  pyplot.show()

def j_theta(theta):
  tmp = (y - numpy.dot(x.T, theta))
  return ((1.0/N) * numpy.dot(tmp.T, tmp))

def batch_gradient_descent():
  theta = [1, 1]
  bf = [theta]

  previous = j_theta(theta)
  current = previous + 1

  i = 1
  while (abs(previous - current) > 10e-6):
    previous = current
    theta = theta + (pas_batch(i) * (1.0/N) * numpy.dot(x, (y - numpy.dot(x.T, theta))))
    bf.append(theta)
    i+=1
    current = j_theta(theta)

  print "Batch theta = ", theta
  return bf

def sigmoid(x):
  a = 1
  b = 0
  den = (1 + (numpy.exp(numpy.dot((-1), (a * x + b)))))
  return (1.0 / den)

def main():
  load_data()
  print_graphs()

if __name__ == '__main__':
  main()
