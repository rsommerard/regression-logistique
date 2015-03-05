# -*- coding: utf8 -*-

import numpy
from matplotlib import pyplot

def load_data():
  """
    Charge les données des fichiers textes.
  """
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
  """
    Pas de la descente de gradient batch pour la dimension des tailles.
  """
  A = 1.0
  B = 1.0
  C = 1.0
  return ((A/(C + (B * val))))

def pas_batch_p(val):
  """
    Pas de la descente de gradient batch pour la dimension des poids.
  """
  A = 1.0
  B = 100.0
  C = 100.0
  return ((A/(C + (B * val))))

def print_graphs():
  """
    Affiche les données sur le graph.
  """
  print 'Légende figure 2 et 3:'
  print '    Les points affichés sur les sigmoides sont des projections des x sur celles-ci.'
  print '    Si le point est au dessus de la courbe de tau, il sera placé dans la catégorie des hommes.'
  print '    Si le point est en dessous de la courbe de tau, il sera placé dans la catégorie des femmes.'
  print '-' * 80

  pyplot.figure(1)
  pyplot.plot(f[:,0], f[:,1], '.', label="f")
  pyplot.plot(h[:,0], h[:,1], '.', label="h")
  pyplot.title('Data')
  pyplot.grid(True)
  pyplot.legend()

  pyplot.figure(2)
  pyplot.plot(f_t[:,0], f_t[:,1], '.', label="f_t")
  pyplot.plot(h_t[:,0], h_t[:,1], '.', label="h_t")
  pyplot.title('Tailles')
  pyplot.grid(True)
  pyplot.plot(f_h_t[:,0], (1 - sigmoid(numpy.dot(x_t.T, batch_theta_t), f_h_t[:,0].mean(0))), '.', label="sigmoide")

  pyplot.plot(h_t[51][0], (1 - sigmoid(numpy.dot(numpy.array([h_t[51][0], 1]).T, batch_theta_t), f_h_t[:,0].mean(0))), 'o g')
  pyplot.plot(h_t[875][0], (1 - sigmoid(numpy.dot(numpy.array([h_t[875][0], 1]).T, batch_theta_t), f_h_t[:,0].mean(0))), 'o g')

  pyplot.plot(f_t[51][0], (1 - sigmoid(numpy.dot(numpy.array([f_t[51][0], 1]).T, batch_theta_t), f_h_t[:,0].mean(0))), 'o b')
  pyplot.plot(f_t[875][0], (1 - sigmoid(numpy.dot(numpy.array([f_t[875][0], 1]).T, batch_theta_t), f_h_t[:,0].mean(0))), 'o b')

  tmp = []
  for i in f_h_t[:,0]:
    tmp.append(tau_t)

  pyplot.plot(f_h_t[:,0], tmp, label="tau")
  pyplot.legend()

  pyplot.figure(3)
  pyplot.plot(f_p[:,0], f_p[:,1], '.', label="f_p")
  pyplot.plot(h_p[:,0], h_p[:,1], '.', label="h_p")
  pyplot.title('Poids')
  pyplot.grid(True)
  pyplot.plot(f_h_p[:,0], (1 - sigmoid(numpy.dot(x_p.T, batch_theta_p), f_h_p[:,0].mean(0))), '.', label="sigmoide")

  pyplot.plot(h_p[51][0], (1 - sigmoid(numpy.dot(numpy.array([h_p[51][0], 1]).T, batch_theta_p), f_h_p[:,0].mean(0))), 'o g')
  pyplot.plot(h_p[875][0], (1 - sigmoid(numpy.dot(numpy.array([h_p[875][0], 1]).T, batch_theta_p), f_h_p[:,0].mean(0))), 'o g')

  pyplot.plot(f_p[51][0], (1 - sigmoid(numpy.dot(numpy.array([f_p[51][0], 1]).T, batch_theta_p), f_h_p[:,0].mean(0))), 'o b')
  pyplot.plot(f_p[875][0], (1 - sigmoid(numpy.dot(numpy.array([f_p[875][0], 1]).T, batch_theta_p), f_h_p[:,0].mean(0))), 'o b')

  tmp = []
  for i in f_h_p[:,0]:
    tmp.append(tau_p)

  pyplot.plot(f_h_p[:,0], tmp, label="tau")
  pyplot.legend()

  pyplot.show()

def j_theta_t(theta):
  """
    Calcul de l'erreur quadratique pour la dimension des tailles.
  """
  tmp = (y_t - numpy.dot(x_t.T, theta))
  return ((1.0/N) * numpy.dot(tmp.T, tmp))

def j_theta_p(theta):
  """
    Calcul de l'erreur quadratique pour la dimension des poids.
  """
  tmp = (y_p - numpy.dot(x_p.T, theta))
  return ((1.0/N) * numpy.dot(tmp.T, tmp))

def batch_gradient_descent_t():
  """
    Calcul de théta par la méthode de descente de gradient batch pour la dimension des tailles.
  """
  theta = numpy.array([1, 1])

  previous = j_theta_t(theta)
  current = previous + 1

  i = 1
  while (abs(previous - current) > 10e-3):
    previous = current
    theta = theta + (pas_batch_t(i) * (1.0/N) * numpy.dot(x_t, (y_t - sigmoid(numpy.dot(x_t.T, theta), f_h_t[:,0].mean(0)))))
    i+=1
    current = j_theta_t(theta)

  print '-' * 80
  print "Théta batch tailles = ", theta
  print '-' * 80
  return theta

def batch_gradient_descent_p():
  """
    Calcul de théta par la méthode de descente de gradient batch pour la dimension des poids.
  """
  theta = numpy.array([1, 1])

  previous = j_theta_p(theta)
  current = previous + 1

  i = 1
  while (abs(previous - current) > 10e-3):
    previous = current
    theta = theta + (pas_batch_p(i) * (1.0/N) * numpy.dot(x_p, (y_p - sigmoid(numpy.dot(x_p.T, theta), f_h_p[:,0].mean(0)))))
    i+=1
    current = j_theta_p(theta)

  print "Théta batch poids = ", theta
  print '-' * 80
  return theta

def sigmoid(a, b=0.0):
  """
    Calcul la sigmoide de a.
  """
  return (1.0 / (1.0 + numpy.exp(-1.0 * (a + b))))

def perf_t(tau):
  """
    Calcul la performance de la classification (taux d'erreur) pour la dimension des tailles.
  """
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
  """
    Calcul la performance de la classification (taux d'erreur) pour la dimension des poids.
  """
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

def find_better_tau_t(start):
  """
    Calcul le meilleur tau qui donne la meilleur performance pour la dimension des tailles.
  """
  better = 0
  better_tau = 0.0
  i = start
  while(i < 1.0):
    res = perf_t(i)
    if(i == start or res < better):
      better = res
      better_tau = i
    i += 0.01

  print 'Recherche du meilleur tau pour la dimension des tailles:'
  print '    - Débute à {0} pour gagner du temps (connaissant la valeur finale).'.format(start)
  print '    - Pas de test de 0.01.'
  print "\n    Meilleur tau: ", better_tau, ' = ', better
  print '-' * 80
  return better_tau

def find_better_tau_p(start):
  """
    Calcul le meilleur tau qui donne la meilleur performance pour la dimension des poids.
  """
  better = 0
  better_tau = 0.0
  i = start
  while(i < 1.0):
    res = perf_p(i)
    if(i == start or res < better):
      better = res
      better_tau = i
    i += 0.01

  print 'Recherche du meilleur tau pour la dimension des poids:'
  print '    - Débute à {0} pour gagner du temps (connaissant la valeur finale).'.format(start)
  print '    - Pas de test de 0.01.'
  print "\n    Meilleur tau: ", better_tau, ' = ', better
  print '-' * 80
  return better_tau

def main():
  """
    Fonction principale du programme.
  """
  global batch_theta_t, batch_theta_p, tau_t, tau_p
  load_data()

  print 'FAA - TP4: Régression logistique'
  print '-' * 80
  print 'Nombre de données: {0}'.format(N)
  print '-' * 80

  batch_theta_t = batch_gradient_descent_t()
  batch_theta_p = batch_gradient_descent_p()

  tau_t = 0.99
  tau_t = find_better_tau_t(0.95)
  print 'Calcul de la performance pour la dimension des tailles:'
  print '    - Valeur de tau: {0}'.format(tau_t)
  print '\n    Performance: {0}'.format(perf_t(tau_t))
  print '-' * 80
  tau_p = 0.97
  tau_p = find_better_tau_p(0.95)
  print 'Calcul de la performance pour la dimension des poids:'
  print '    - Valeur de tau: {0}'.format(tau_p)
  print '\n    Performance: {0}'.format(perf_p(tau_p))
  print '-' * 80

  print_graphs()

if __name__ == '__main__':
  main()
