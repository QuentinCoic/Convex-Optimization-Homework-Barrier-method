# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 19:29:09 2017

@author: Quentin
"""

import numpy as np
import matplotlib.pyplot as plt
from fonctions import *

"""
    On fait la partie sur le "Iris dataset" avec la "damped Newton method"
"""

X = np.loadtxt("iris.data",delimiter=',',skiprows=50,usecols=(0,1,2,3))
n,d = X.shape
moyennes = np.mean(X,axis=0)
X = X-moyennes # On centre notre dataset
y = np.ones(n)
y[:int(n/2)] = -y[:int(n/2)] # On donne le label -1 à classe "Iris-versicolor" et 1 à la classe "Iris-virginica"

n0 = int(n*8/10)
n1 = n-n0
l0 = np.random.choice(range(n),n0,False) # On choisit aléatoirement 80% de notre dataset pour faire la classification
l1 = np.array([i for i in range(n) if not i in l0])
X0 = X[l0]
X1 = X[l1] # Les données sur lesquelles on va tester la performance de notre classification
y0 = y[l0]
y1 = y[l1]

def dampedClassification(mu,tol) :
    """
        Retourne la liste des tau utilisés, la liste des erreurs correspondantes
        pour la classification des donnés de X1 ainsi que la liste des 
        classifiactions obtenues de X1 pour chaque valeur de tau.
    """
    tau = 0
    listau = []
    t = 1
    erreurs = []
    classes = []
    
    for i in range(101) : # On fait varier tau de 0.1 à 10 par pas de 0.1
        tau += 0.1
        listau.append(tau)
        Q,p,A,b = transform_svm_primal(tau,X0,y0)
        Q2,p2,A2,b2 = transform_svm_dual(tau,X0,y0)
        x_0 = np.concatenate((np.zeros(d),2*np.ones(n0))) # Point strictemet faisable pour le primal
        x2_0 = np.ones(n0)/(2*tau*n0) # Point strictemet faisable pour le dual
        eta = quad(x_0,Q,p) + quad(x2_0,Q2,p2) # Duality gap initial 
        t = len(b)/eta # t initial 
        x_sol,xhist,dgap = barr_method(Q,p,A,b,x_0,mu,tol,t)
        w = x_sol[:d]
        c = np.dot(X1,w)
        classes.append(np.zeros(n1))
        classes[i][c>0] = 1 # On sépare les données qui sont de chaque côté de l'hyperplan w.T.x = 0
        classes[i][c<=0] = -1 # Label 1 si w.T*x > 0 et label -1 sinon
        erreurs.append(np.sum(y1!=classes[i])/n1) # Le taux d'erreurs de classification sur X1
    
    plt.plot(listau,erreurs)
    plt.title("Taux d'erreurs sur la partie de l'échantillon non utilisée pour l'apprentissage - Damped newton")
    plt.xlabel("Tau")
    plt.ylabel("Taux d'erreurs")
    plt.show()
    plt.close()
    return listau,erreurs,classes

"""tol = 1e-3
mu = 10
listau,erreurs,classes = dampedClassification(mu,tol)"""

def genDataSetDamped(tau,mu,tol) :
    """
        Génère aléatoirement un nouveau "Iris dataset".
    """
    Q,p,A,b = transform_svm_primal(tau,X0,y0)
    Q2,p2,A2,b2 = transform_svm_dual(tau,X0,y0)
    x_0 = np.concatenate((np.zeros(d),2*np.ones(n0)))
    x2_0 = np.ones(n0)/(2*tau*n0)
    eta = quad(x_0,Q,p) + quad(x2_0,Q2,p2)
    t = len(b)/eta
    
    x_sol,xhist,dgap = barr_method(Q,p,A,b,x_0,mu,tol,t)
    w = x_sol[:d]
    
    Z = np.random.multivariate_normal(np.zeros(d),np.cov(X,rowvar=False),n) # On génère un dataset centré en utilisant la matrice de variance-covariance empirique de X 
    
    c = np.dot(Z,w)
    classes = np.zeros(n)
    classes[c>0] = 1 # On donne les labels aux points du nouveau dataset grâce à l'hyperplan séparateur obtenu via X0
    classes[c<=0] = -1
    
    Z = Z + moyennes #On "décentre" les points 
    return Z,classes

"""tol = 1e-3
tau = 0.1
mu = 10
Z,classes = genDataSetDamped(tau,mu,tol)"""

def dampedDgapVIterationsPrimal(tau,tol) :
    """
        Retourne pour le primal les points optimaux selon la méthode log-barrière (utilisant 
        la "damped Newton method") et l'historique de tous les points parcourus
        pour mu = 2,15,50 et 100.
        Trace la courbe du "duality gap versus iterations" pour chaque mu.
    """
    Q,p,A,b = transform_svm_primal(tau,X0,y0)
    Q2,p2,A2,b2 = transform_svm_dual(tau,X0,y0)
    x_0 = np.concatenate((np.zeros(d),2*np.ones(n0)))
    x2_0 = np.ones(n0)/(2*tau*n0)
    eta = quad(x_0,Q,p) + quad(x2_0,Q2,p2)
    m = len(b)
    
    mus = [2,15,50,100]
    xsol = []
    xhist = []
    dgap = []
    pl=[]
    
    for i,mu in enumerate(mus) : 
        t = m/eta
        tmp1,tmp2,tmp3 = barr_method(Q,p,A,b,x_0,mu,tol,t)
        xsol.append(tmp1)
        xhist.append(tmp2)
        dgap.append(tmp3)
        pl.append(plt.semilogy(dgap[i],label='mu={}'.format(mu)))
    
    plt.title("Duality gap versus cumulative number of Newton steps - Primal - Damped newton")
    plt.xlabel("Damped newton iterations")
    plt.ylabel("Duality gap")
    plt.legend()
    plt.show()
    plt.close()
    return xsol,xhist

"""tol = 1e-3
tau = 0.1
xsol,xhist = dampedDgapVIterationsPrimal(tau,tol)"""
    
def dampedDgapVIterationsDual(tau,tol) :
    """
        Retourne pour le dual les points optimaux selon la méthode log-barrière (utilisant 
        la "damped Newton method") et l'historique de tous les points parcourus
        pour mu = 2,15,50 et 100.
        Trace la courbe du "duality gap versus iterations" pour chaque mu.
    """
    Q,p,A,b = transform_svm_primal(tau,X0,y0)
    Q2,p2,A2,b2 = transform_svm_dual(tau,X0,y0)
    x_0 = np.concatenate((np.zeros(d),2*np.ones(n0)))
    x2_0 = np.ones(n0)/(2*tau*n0)
    eta = quad(x_0,Q,p) + quad(x2_0,Q2,p2)
    m = len(b)
    
    mus = [2,15,50,100]
    xsol = []
    xhist = []
    dgap = []
    pl=[]
    
    for i,mu in enumerate(mus) : 
        t = m/eta
        tmp1,tmp2,tmp3 = barr_method(Q2,p2,A2,b2,x2_0,mu,tol,t)
        xsol.append(tmp1)
        xhist.append(tmp2)
        dgap.append(tmp3)
        pl.append(plt.semilogy(dgap[i],label='mu={}'.format(mu)))
    
    plt.title("Duality gap versus cumulative number of Newton steps - Dual - Damped newton")
    plt.xlabel("Damped newton iterations")
    plt.ylabel("Duality gap")
    plt.legend()
    plt.show()
    plt.close()    
    return xsol,xhist

"""tol = 1e-3
tau = 0.1
xsol, xhist = dampedDgapVIterationsDual(tau,tol)"""





    
