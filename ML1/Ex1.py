#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 15:04:43 2019

@author: hao
"""

import prtools as pr
import numpy as np
import matplotlib.pyplot as plt

# --------------EX 1.3------------------------
# 1.6
x = np.array([[0.7, 0.3, 0.2],[2.1, 4.5, 0]])
#print(x)
mean0 = np.mean(x)
#print(mean0)
std = np.std(x)
#print(std)
mean1 = np.mean(x, axis = 0)
#print(mean1)
mean2 = np.mean(x, axis = 1)
#print(mean2)

# 1.7
#plt.scatter(x[:,0], x[:,1])

# 1.8
lab = np.matrix([[1, 2], [1, 2], [1, 2]]).T
#print(lab)
a = pr.prdataset(x, lab)
#print(a)

# 1.9
b = pr.boomerangs(100)
#pr.scatterd(b)
#pr.scatterd(b[:,[1, 2]]) 




#-----------------EX 1.4 -------------------------
c = pr.gendatb()    # c = pr.gendatb()
w = pr.nmc(c)       # w = pr.nmc()  
#print(w)        

d = c*w             #w.train(c)
#print(d)            

lab2 = d *pr.labeld()   #lab2 = w.eval(c)  evaluate
#print(lab2)

e = d * pr.testc()  # e = pr.testc(lab2)
#print(e)

w2 = pr.svc(c, ('rbf', 4.5, 1)) #rbf is kernel, kernal params, The tradeoff between the margin and training hinge loss is defined by parameter C

a1 = pr.gendath()
w1 = pr.parzenc(a1)
#pr.scatterd(a1)
#pr.plotc(w1)

# 1.10
a2 = pr.gendatd(100)
#pr.scatterd(a2)
w2 = pr.ldc(a2) #nmc, naviebc(for many features), stumpc, svc is far incorect; dectreec is overfit
#pr.plotc(w2)



#-------------------------EX 1.5 --------------------------------------
    
   # Generation of a simple classification data.

   #    A = gendats(N,DIM,DELTA)

   # Generate a two-class dataset A from two DIM-dimensional Gaussian
   # distributions, containing N samples. Optionally, the mean of the
   # first class can be shifted by an amount of DELTA.
    
a3 = pr.gendats([20, 20], 1, 8);  # [20, 20]
#print("before : ", a3)
#pr.scatterd(a3)
h = 1;
a3 = pr.prdataset(+a3) #[40]
#print(np.array(a3))
w3 = pr.parzenm(a3, h)
#pr.scatterd(a3);
#pr.plotm(w3)

# 1.12
a4 = pr.gendats([20, 20], 1, 8)
a4 = pr.prdataset(+a4)
hs = [0.01, 0.05, 0.1, 0.25, 0.5, 1, 1.5, 2, 3, 4, 5]
LL = np.zeros(len(hs))
for i in range(len(hs)):
    w4 = pr.parzenm(a4, hs[i])
    LL[i] = np.sum(np.log(+(a4*w4)))
#plt.plot(hs, LL);

# 1.13
a5 = pr.gendats([20, 20], 1, 8)
a5 = pr.prdataset(+a5)
[trn, tst] = pr.gendat(a5, 0.5)
hs2 = [0.01, 0.05, 0.1, 0.25, 0.5, 1, 1.5, 2, 3, 4, 5]
Ltrn = np.zeros(len(hs2))
Ltst = np.zeros(len(hs2))
for i in range(len(hs2)):
    w5 = pr.parzenm(trn, hs2[i])
    Ltrn[i] = np.sum(np.log(+(trn*w5)))
    Ltst[i] = np.sum(np.log(+(tst*w5)))
plt.plot(hs2, Ltrn, 'b-')
plt.plot(hs2, Ltst, 'r-')    
