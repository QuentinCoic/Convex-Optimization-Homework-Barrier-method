# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 15:38:30 2017

@author: Quentin
"""

import numpy as np

def phi(x,t,Q,p,A,b) :
    return t*((1/2)*np.dot(x,np.dot(Q,x)) + np.dot(p,x)) - np.sum(np.log(b-np.dot(A,x)))

def grad(x,t,Q,p,A,b) :
    return t*(np.dot(Q,x) + p) + np.sum(A.transpose()*(1/(b-np.dot(A,x))),axis=1)

def hess(x,t,Q,p,A,b) :
    Abis = A.transpose()*(1/(b-np.dot(A,x)))
    return t*Q + np.dot(Abis,Abis.transpose())

def dampedNewtonStep(x,g,h):
    """
        Retourne le point suivant de l'algorithme damped Newton à partir du 
        point x ainsi que l'écart estimé entre phit(x) et la valeur optimale
        i.e. le "Newton decrement" au carré divisé par 2.
    """
    L = np.linalg.cholesky(h(x))
    L_inv = np.linalg.inv(L)
    g_x = g(x)
    ndec = np.linalg.norm(np.dot(L_inv,g_x))
    return x-np.dot(L_inv.transpose(),np.dot(L_inv,g_x))/(1+ndec),ndec**2/2
    
def dampedNewton(x0,g,h,tol) :
    """
        Retourne le point optimal selon l'algorithme damped Newton avec une
        tolérance "tol" et en partant de x0. Retourne également l'historique de
        tous les points parcourus durant l'algorithme ainsi qu'un compteur 
        d'itérations qui nous servira à tracer le "duality gap versus 
        iterations" de la méthode log-barrière.
    """
    tol = min(tol,(3-np.sqrt(5))/2)
    xstar = x0
    xhist = [xstar]
    compteur = 1
    xnew,gap = dampedNewtonStep(xstar,g,h)
    while gap>tol :
        xstar = xnew
        xhist.append(xstar)
        xnew,gap = dampedNewtonStep(xstar,g,h)
        compteur += 1
    return xstar,xhist,compteur
    
def backls(x,f,g,A,b,nstep,alpha,beta) :
    """
        Effectue une "backtracking line search sur la ligne x + t*nstep où 
        nstep correspond au "Newton step" en x avec les paramètres alpha et 
        beta. On effectue une opération préliminaire avant la "line search" 
        afin de s'assurer que l'on est bien dans le domaine de phit i.e. que 
        b(i) - a(i).T*x est bien >0. On prend pour cela A et b en entrée.
    """
    t = 1
    ineq = b-np.dot(A,x+t*nstep)
    while len(ineq[ineq<=0])>0 :
        t = beta*t
        ineq = b-np.dot(A,x+t*nstep)
    while f(x+t*nstep)>f(x)+alpha*t*np.dot(g(x),nstep) :
        t = beta*t
    return t

def newtonLSStepDec(x,g,h):
    """
        Retourne le "Newton step" en x aisi que le "Newton decrement" au carré
        en x.
    """
    L = np.linalg.cholesky(h(x))
    L_inv = np.linalg.inv(L)
    g_x = g(x)
    return -np.dot(L_inv.transpose(),np.dot(L_inv,g_x)),np.linalg.norm(np.dot(L_inv,g_x))**2   

def newtonLS(x0,f,g,h,A,b,tol,alpha,beta) :
    """
        Retourne le point optimal selon l'algorithme Newton LS avec une
        tolérance "tol" et en partant de x0. Retourne également l'historique de
        tous les points parcourus durant l'algorithme ainsi qu'un compteur 
        d'itérations qui nous servira à tracer le "duality gap versus 
        iterations" de la méthode log-barrière. Prend en entrée A,b, alpha et
        beta pour la "backtracking line search".
    """
    xstar = x0
    xhist = [xstar]
    compteur = 1
    nstep,ndec = newtonLSStepDec(xstar,g,h)
    while ndec/2>tol :
        xstar = xstar + nstep*backls(xstar,f,g,A,b,nstep,alpha,beta)
        xhist.append(xstar)
        nstep,ndec = newtonLSStepDec(xstar,g,h)
        compteur += 1
    return xstar,xhist,compteur

def transform_svm_primal(tau,X,y) :
    n,d = X.shape
    K = np.zeros((n,d))
    u = np.ones(n)
    I = np.eye(n)
    Q = np.concatenate((np.concatenate((np.eye(d),K)),np.zeros((d+n,n))),axis=1)
    p = np.concatenate((np.zeros(d),u/(n*tau)))
    A = np.concatenate((np.concatenate((-np.dot(np.diag(y),X),K)),np.concatenate((-I,-I))),axis=1)
    b = np.concatenate((-u,np.zeros(n)))
    return Q,p,A,b

def transform_svm_dual(tau,X,y) :
    n = X.shape[0]
    K = np.dot(np.diag(y),X)
    Q = np.dot(K,K.transpose())
    p = -np.ones(n)
    K = np.eye(n)
    A = np.concatenate((K,-K))
    b = np.concatenate((-p/(n*tau),np.zeros(n)))
    return Q,p,A,b

def barr_method(Q,p,A,b,x_0,mu,tol,t) :
    """
        Retourne le point optimal selon la méthode log-barrière (utilisant la
        "damped Newton method") avec un tolérance "tol", un paramètre mu et en 
        partant de x_0 et t. Retourne également l'historique de tous les points
        de la "damped Newton method" parcourus durant l'algorithme ainsi que qu'un 
        vecteur dgap qui nous permettra de tracer le "duality gap versus 
        iterations".
    """
    g = lambda x: grad(x,t,Q,p,A,b)
    h = lambda x: hess(x,t,Q,p,A,b)
    x_sol,xhist,compteur = dampedNewton(x_0,g,h,tol)
    xhist = [xhist]
    xhistbis = []
    m = len(b)
    dgap = (m/t)*np.ones(compteur)
    while m/t>tol :
        t = t*mu
        g = lambda x: grad(x,t,Q,p,A,b)
        h = lambda x: hess(x,t,Q,p,A,b)
        x_sol,xhistbis,compteur = dampedNewton(x_sol,g,h,tol)
        xhist.append(xhistbis)
        dgap = np.concatenate((dgap,(m/t)*np.ones(compteur)))
    return x_sol,xhist,dgap

def barr_method2(Q,p,A,b,x_0,mu,tol,t,alpha,beta) :
    """
        Retourne le point optimal selon la méthode log-barrière (utilisant la
        "LS Newton method") avec un tolérance "tol", un paramètre mu et en 
        partant de x_0 et t. Retourne également l'historique de tous les points
        de la "LS Newton method" parcourus durant l'algorithme ainsi que qu'un 
        vecteur dgap qui nous permettra de tracer le "duality gap versus 
        iterations".
    """
    f = lambda x: phi(x,t,Q,p,A,b)
    g = lambda x: grad(x,t,Q,p,A,b)
    h = lambda x: hess(x,t,Q,p,A,b)
    x_sol,xhist,compteur = newtonLS(x_0,f,g,h,A,b,tol,alpha,beta)
    xhist = [xhist]
    xhistbis = []
    m = len(b)
    dgap = (m/t)*np.ones(compteur)
    while m/t>tol :
        t *= mu
        f = lambda x: phi(x,t,Q,p,A,b)
        g = lambda x: grad(x,t,Q,p,A,b)
        h = lambda x: hess(x,t,Q,p,A,b)
        x_sol,xhistbis,compteur = newtonLS(x_sol,f,g,h,A,b,tol,alpha,beta)
        xhist.append(xhistbis)
        dgap = np.concatenate((dgap,(m/t)*np.ones(compteur)))
    return x_sol,xhist,dgap

def quad(x,Q,p) :
    """
    Retourne la valeur d'une fonction quadratique de la forme (1/2)x.T*Q*x + 
    p.T*x en x. On s'en servira pour calculer la valeur initiale de t dans la
    méthode log-barrière.
    """
    return (1/2)*np.dot(x,np.dot(Q,x)) + np.dot(p,x)
    

    
    