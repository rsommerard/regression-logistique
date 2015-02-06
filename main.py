# -*- coding: utf8 -*-

import numpy
from matplotlib import pyplot
import random

def load_data():
  global x, y, N, t, p
  t = numpy.loadtxt('data/taillepoids_f.txt')
  N = len(t)
  x = t
  p = numpy.loadtxt('data/taillepoids_h.txt')
  y = p

def print_graphs():
  pyplot.figure(1)
  pyplot.plot(t, '.')

  pyplot.figure(2)
  pyplot.plot(p, '.')

  pyplot.show()

def main():
  load_data()
  print_graphs()

if __name__ == '__main__':
  main()
  
