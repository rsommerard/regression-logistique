# -*- coding: utf8 -*-

import numpy
from matplotlib import pyplot

def load_data():
  global f, h, f_t, h_t, f_p, h_p, f_h_t, f_h_p, N, x_t, y_t, x_p, y_p
  f = numpy.loadtxt('data/taillepoids_f.txt')
  h = numpy.loadtxt('data/taillepoids_h.txt')

  f_t = f.copy()
  h_t = h.copy()
  f_t[:,1] = numpy.zeros(len(f_t))
  h_t[:,1] = numpy.ones(len(h_t))
  f_h_t = numpy.concatenate((f_t, h_t), axis=0)

  f_p = f.copy()
  h_p = h.copy()
  f_p[:,0] = f_p[:,1]
  h_p[:,0] = h_p[:,1]
  f_p[:,1] = numpy.zeros(len(f_p))
  h_p[:,1] = numpy.ones(len(h_p))
  f_h_p = numpy.concatenate((f_p, h_p), axis=0)

  N = len(f_h_t)

  x_t = numpy.vstack((f_h_t[:,0], numpy.ones(N)))
  y_t = f_h_t[:,1]

  x_p = numpy.vstack((f_h_p[:,0], numpy.ones(N)))
  y_p = f_h_p[:,1]

def pas_batch_t(val):
  A = 1.0
  B = 1.0
  C = 1.0
  return ((A/(C + (B * val))))

def pas_batch_p(val):
  A = 1.0
  B = 100.0
  C = 100.0
  return ((A/(C + (B * val))))

def print_graphs():
  pyplot.figure(1)
  pyplot.plot(f[:,0], f[:,1], '.', label="f")
  pyplot.plot(h[:,0], h[:,1], '.', label="h")
  pyplot.legend()

  pyplot.figure(2)
  pyplot.plot(f_t[:,0], f_t[:,1], '.', label="f_t")
  pyplot.plot(h_t[:,0], h_t[:,1], '.', label="h_t")
  pyplot.plot(f_h_t[:,0], (1 - sigmoid(numpy.dot(x_t.T, batch_theta_t), f_h_t[:,0].mean(0))), '.')

  pyplot.plot(h_t[51][0], (1 - sigmoid(numpy.dot(numpy.array([h_t[51][0], 1]).T, batch_theta_t), f_h_t[:,0].mean(0))), 'o g')
  pyplot.plot(h_t[875][0], (1 - sigmoid(numpy.dot(numpy.array([h_t[875][0], 1]).T, batch_theta_t), f_h_t[:,0].mean(0))), 'o g')

  pyplot.plot(f_t[51][0], (1 - sigmoid(numpy.dot(numpy.array([f_t[51][0], 1]).T, batch_theta_t), f_h_t[:,0].mean(0))), 'o b')
  pyplot.plot(f_t[875][0], (1 - sigmoid(numpy.dot(numpy.array([f_t[875][0], 1]).T, batch_theta_t), f_h_t[:,0].mean(0))), 'o b')

  tmp = []
  for i in f_h_t[:,0]:
    tmp.append(tau_t)

  pyplot.plot(f_h_t[:,0], tmp)
  pyplot.legend()

  pyplot.figure(3)
  pyplot.plot(f_p[:,0], f_p[:,1], '.', label="f_p")
  pyplot.plot(h_p[:,0], h_p[:,1], '.', label="h_p")
  pyplot.plot(f_h_p[:,0], (1 - sigmoid(numpy.dot(x_p.T, batch_theta_p), f_h_p[:,0].mean(0))), '.')

  pyplot.plot(h_p[51][0], (1 - sigmoid(numpy.dot(numpy.array([h_p[51][0], 1]).T, batch_theta_p), f_h_p[:,0].mean(0))), 'o g')
  pyplot.plot(h_p[875][0], (1 - sigmoid(numpy.dot(numpy.array([h_p[875][0], 1]).T, batch_theta_p), f_h_p[:,0].mean(0))), 'o g')

  pyplot.plot(f_p[51][0], (1 - sigmoid(numpy.dot(numpy.array([f_p[51][0], 1]).T, batch_theta_p), f_h_p[:,0].mean(0))), 'o b')
  pyplot.plot(f_p[875][0], (1 - sigmoid(numpy.dot(numpy.array([f_p[875][0], 1]).T, batch_theta_p), f_h_p[:,0].mean(0))), 'o b')

  tmp = []
  for i in f_h_p[:,0]:
    tmp.append(tau_p)

  pyplot.plot(f_h_p[:,0], tmp)
  pyplot.legend()

  pyplot.show()

def j_theta_t(theta):
  tmp = (y_t - numpy.dot(x_t.T, theta))
  return ((1.0/N) * numpy.dot(tmp.T, tmp))

def j_theta_p(theta):
  tmp = (y_p - numpy.dot(x_p.T, theta))
  return ((1.0/N) * numpy.dot(tmp.T, tmp))

def batch_gradient_descent_t():
  theta = numpy.array([1, 1])

  previous = j_theta_t(theta)
  current = previous + 1

  i = 1
  while (abs(previous - current) > 10e-3):
    previous = current
    theta = theta + (pas_batch_t(i) * (1.0/N) * numpy.dot(x_t, (y_t - sigmoid(numpy.dot(x_t.T, theta), f_h_t[:,0].mean(0)))))
    i+=1
    current = j_theta_t(theta)

  print "Batch theta = ", theta
  return theta

def batch_gradient_descent_p():
  theta = numpy.array([1, 1])

  previous = j_theta_p(theta)
  current = previous + 1

  i = 1
  while (abs(previous - current) > 10e-3):
    previous = current
    theta = theta + (pas_batch_p(i) * (1.0/N) * numpy.dot(x_p, (y_p - sigmoid(numpy.dot(x_p.T, theta), f_h_p[:,0].mean(0)))))
    i+=1
    current = j_theta_p(theta)

  print "Batch theta = ", theta
  return theta

def sigmoid(a, b=0.0):
  return (1.0 / (1.0 + numpy.exp(-1.0 * (a + b))))

def perf_t(tau):
  res = []
  for i in range(0, N):
    if ((1 - sigmoid(numpy.dot(numpy.array([f_h_t[i][0], 1]).T, batch_theta_t), f_h_t[:,0].mean(0))) < tau):
      res.append(0)
    else:
      res.append(1)

  sum = 0
  for i in range(0, N):
    sum += abs(f_h_t[i][1] - res[i])

  return (1.0/N) * sum

def perf_p(tau):
  res = []
  for i in range(0, N):
    if ((1 - sigmoid(numpy.dot(numpy.array([f_h_p[i][0], 1]).T, batch_theta_p), f_h_p[:,0].mean(0))) < tau):
      res.append(0)
    else:
      res.append(1)

  sum = 0
  for i in range(0, N):
    sum += abs(f_h_p[i][1] - res[i])

  return (1.0/N) * sum

def find_better_tau_t():
  better = 0
  better_tau = 0.0
  i = 0.0
  while(i < 1.0):
    res = perf_t(i)
    if(i == 0.0 or res < better):
      better = res
      better_tau = i
    i += 0.01

  print 'better tau t: ', better_tau, ' = ', better
  return better_tau

def find_better_tau_p():
  better = 0
  better_tau = 0.0
  i = 0.0
  while(i < 1.0):
    res = perf_p(i)
    if(i == 0.0 or res < better):
      better = res
      better_tau = i
    i += 0.01

  print 'better tau p: ', better_tau, ' = ', better
  return better_tau

def main():
  global batch_theta_t, batch_theta_p, tau_t, tau_p
  load_data()

  batch_theta_t = batch_gradient_descent_t()
  batch_theta_p = batch_gradient_descent_p()

  #tau_t = find_better_tau_t()
  tau_t = 0.99
  #tau_p = find_better_tau_p()
  tau_p = 0.97

  print_graphs()

if __name__ == '__main__':
  main()
