# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 17:52:50 2017

@author: Quentin
"""

import numpy as np 
import matplotlib.pyplot as plt
from fonctions import *

"""
    On teste sur l'exemple très simple de minimisation (1/2)x**2 + x sous la 
    contrainte que x>=0 nos algorithmes pour la "LS Newton method".
"""

def LSDgapVIterations(tol,alpha,beta) :
    """
        Retourne les points optimaux selon la méthode log-barrière (utilisant 
        la "LS Newton method") et l'historique de tous les points parcourus
        pour mu = 2,15,50 et 100.
        Trace la courbe du "duality gap versus iterations" pour chaque mu.
    """
    Q = np.array([[1]])
    p = np.array([1])
    A = np.array([[-1]])
    b = np.array([0])
    x_0 = np.array([1]) # Point strictement faisable pour les problèmes primal et dual
    eta = quad(x_0,Q,p) - quad(x_0,-1,1) + 0.5 # Duality gap initial (pour calculer le t initial)
    
    mus = [2,15,50,100]
    xsol = []
    xhist = []
    dgap = []
    pl=[]
    
    for i,mu in enumerate(mus) : 
        t = 1/eta
        tmp1,tmp2,tmp3 = barr_method2(Q,p,A,b,x_0,mu,tol,t,alpha,beta)
        xsol.append(tmp1)
        xhist.append(tmp2)
        dgap.append(tmp3)
        pl.append(plt.semilogy(dgap[i],label='mu={}'.format(mu)))
    
    plt.title("Duality gap versus cumulative number of Newton steps - LS newton")
    plt.xlabel("LS iterations")
    plt.ylabel("Duality gap")
    plt.legend()
    plt.show()
    plt.close()
    return xsol,xhist

tol = 1e-3
alpha = 0.01
beta = 0.5
xsol,xhist = LSDgapVIterations(tol,alpha,beta)