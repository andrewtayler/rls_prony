#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 14:43:08 2021

@author: andrew
"""

import numpy as np
import matplotlib.pyplot as mpl
import math
from RLS_Prony_v2 import RLS_Prony
import scipy.io as sio

mat = sio.loadmat('shipData_1.mat')
t = np.asarray(mat["t"]).flatten()
data = np.asarray(mat["Z"]).flatten()
N = len(t)

# f = lambda x: math.sin(2*math.pi*x)# + math.sin(2*math.pi*0.01*x)
# T = 5;
# N = 1000 # Number of points (including 0)
# t = np.arange(0,T,T/N)
# data = np.array([f(i) for i in t])

M = 30 # Number of coefficients including x**0
lam = 0.6 # Forgetting factor
Ts = t[1]-t[0]
prony = RLS_Prony(M,lam,Ts)

e = []
yn = 0
yt = []

tp = 0

data_trim = data[0:-tp]

for i,y in enumerate(data):
    prony.updateLPM(y)
    if prony.num_obs > prony.M:
        prony.updateProny(t[i])
        y_est = prony.get_est(t[i])
        yt.append(y_est)
    else:
        yt.append(0)
    e.append(prony.get_error())
    
# for i,y in enumerate(t[-tp:]):
#     yt.append(prony.get_est(y))

print(f'Coefficients are: {prony.A}')
print(f'Roots are: {prony.get_roots()}')

fig_2 = mpl.figure()
er = fig_2.add_axes([0,0,1,1])
#er.plot(t,np.real(e))
er.plot(t,e)
er.set_title('Error')
er.set_xlim([0,200])
#er.set_ylim([-10,10])

fig, yp = mpl.subplots(1, 1, tight_layout=True)
fig.set_size_inches(8,4)
yp.plot(t,data,linewidth=2)
yp.plot(t,np.real(yt),'--',linewidth=3)
yp.set_xlim([0,200])
yp.set_ylabel('Heave (m)')
yp.set_xlabel('Time (s)')
yp.legend(['Simulated Heave Motion', 'Prony Approximation'])
fig.savefig('pronyTest.eps')
#yp.set_ylim([-2,2])


#%%

