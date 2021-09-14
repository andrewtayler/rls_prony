#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 14:43:08 2021

@author: andrew
"""

import numpy as np
import matplotlib.pyplot as mpl
from RLS_Prony import RLS_Prony
import scipy.io as sio
import math

mat = sio.loadmat('shipData_4.mat')
t = np.asarray(mat["t"]).flatten()
data = np.asarray(mat["Z"]).flatten()
N = len(t)

# f = lambda x: math.sin(2*math.pi*x)# + math.sin(2*math.pi*0.01*x)
# T = 5;
# N = 1000 # Number of points (including 0)
# t = np.arange(0,T,T/N)
# data = np.array([f(i) for i in t])

M = 13 # Number of coefficients including x**0
lam = 0.8 # Forgetting factor
Ts = t[1]-t[0]
prony = RLS_Prony(M,lam,Ts)

e = []
yn = 0
yt = []

tp = 0

N_1 = 200

for i,y in enumerate(data[0:N_1]):
    prony.update(y)

    e.append(prony.get_error())
    
    yt.append(prony.get_est())
    

print(f'Coefficients are: {prony.A}')
print(f'Roots are: {prony.get_roots()}')


#%% Plotting

fig_2 = mpl.figure()
er = fig_2.add_axes([0,0,1,1])
#er.plot(t,np.real(e))
er.plot(t[0:N_1],e)
er.set_title('Error')
#er.set_ylim([-10,10])
#er.set_xlim([0,15])

#%%

fig, yp = mpl.subplots(1, 1, tight_layout=True)
fig.set_size_inches(8,4)
yp.plot(t[0:N_1],data[0:N_1],linewidth=2)
yp.plot(t[0:N_1],np.real(yt),'--',linewidth=2)
yp.set_ylabel('Heave (m)')
yp.set_xlabel('Time (s)')
yp.legend(['Simulated Heave Motion', 'Prony Approximation'])
#fig.savefig('pronyTest.eps')
# yp.set_ylim([-0.02,0.02])
#yp.set_xlim([150,300])


#%%
ye = []

for dt in range(len(t[N_1:])):
    ye.append(prony.predict(dt*Ts))

yp.plot(t[N_1:],data[N_1:],linewidth=1)
yp.plot(t[N_1:],np.real(ye),'--',linewidth=2)  
yp.set_ylim([-0.02,0.02])
# yp.set_ylim([-1,1])
# yp.set_xlim([0,100])

