#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 13:46:03 2021

@author: andrew
"""

import numpy as np
import numpy.polynomial.polynomial as poly
import sys
import math
from scipy.linalg import lstsq

class RLS_Prony:

    def __init__(self, M, lam, Ts):
        '''
        M: number of variables including constant (model order)
        lam: forgetting factor
        '''
        
        self.M = M # Number of variables
        self.lam = lam # Forgetting Factor
        self.Ts = Ts # Sampling time
        
        # Initialise error covariance
        self.P1 = np.matrix(np.identity(self.M))
        
        # Initialise errors
        self.e1 = 0
        self.sse = 0
        
        self.A = np.matrix(np.zeros([M,1]))
        self.r = np.array(np.zeros([M,1]))
        self.H = np.matrix(np.zeros([M,1]))  
        
        self.obs_buffer = [0]
        self.obs_limit = self.M
        
        # Fill with buffer with zeros up to the model order
        while(len(self.obs_buffer) < (self.M)):
            self.obs_buffer.insert(0,0)  
        
        self.num_obs = 0
        
        self.y_est = 0
        
    def update(self,y):
        
        # TODO
        # Test SSE with current model order
        # Increase model order by one if enough error
        
        # Update
        self.lpm(y)
 
        # Slide the input window along
        self.obs_buffer.insert(0,-y)
        
        if len(self.obs_buffer) > self.obs_limit:
            self.obs_buffer.pop()
     
        self.num_obs += 1
               
        self.prony()
        
        
    def lpm(self,y):
        lam_inv = 1/self.lam
        
        # Get past inputs [y(N-2) y(N-3) ... y(N-M)]
        un = np.asmatrix(self.obs_buffer[0:self.M]).T 
        
        Pn_1 = self.P1
        
        # Gain
        K = (lam_inv*Pn_1@un)/(1+lam_inv*un.T@Pn_1@un)

        # Test to see how the old estimate compares to the new data
        # y is the new input y(N-1)
        self.e1 = np.asscalar(y - self.A.T@un)
    
        # New coefficients are the old + the gain * the error
        self.A = self.A + K*self.e1 
    
        # Update inverse
        self.P1 = lam_inv*Pn_1 - lam_inv*K@un.T@Pn_1
   
    def prony(self): 
        y = np.flip(-np.array(self.obs_buffer))
        
        self.r = self.get_roots()
            
        # Initialise the Vandermonde matrix
        uz = np.asmatrix(np.zeros([len(self.obs_buffer),len(self.r)],dtype=complex))
        
        for i in range(len(self.obs_buffer)):
            for j in range(len(self.r)):
                uz[i,j] = self.r[j]**(i)
            
        # Replace Inf values with max     
        uz = self.remove_inf(uz)
        
        self.H = np.array(lstsq(uz,y)[0])      
        return
    
    def get_prony(self):
        
        rts = self.get_roots()     
        A,B = self.get_dom_poles(rts)
        
        return A,B
    
    def predict(self,t): 
        
        y_est = float(0)
        k = (self.obs_limit)*self.Ts
        
        A,B = self.get_prony()
        
        for j in range(len(A)):
            y_est = y_est + A[j]*np.exp(B[j]*(t+k))
        return y_est
        

    def get_error(self):
        e = -self.obs_buffer[0] - self.predict(-1*self.Ts)
        self.sse += e**2
        return e
    
    def get_roots(self):
        # Find the roots of the characteristic equation
        c = np.asarray(self.A).reshape(-1)
        c = np.insert(c,0,1)
        r = poly.polyroots(c)
        
        return r

    def remove_inf(self,mat):
        mat_r = np.asarray(np.real(mat))
        mat_i = np.asarray(np.imag(mat))
        
        mat_r[np.isinf(mat_r)] = sys.float_info.max*np.sign(mat_r[np.isinf(mat_r)])
        mat_i[np.isinf(mat_i)] = sys.float_info.max*np.sign(mat_i[np.isinf(mat_i)])
        
        return mat_r + 1j*mat_i
    
    def get_dom_poles(self,rts):
        Lp = math.inf
        
        ai = np.log(np.abs(rts))/self.Ts
        bi = (np.arctan(np.imag(rts)/np.real(rts)))/self.Ts
        lam_i = ai + (bi)*1j
        
        # Remove non-dominant poles
        try:
            d = lam_i[np.real(lam_i) < 0]
            d = np.max(np.real(d))
            Lp = 1e4*np.abs(np.real(d))# Set poles threshold
        except:
            Lp = math.inf
        
        A = []
        B = []
        
        for i in range(len(self.H)):           
            if np.abs(np.real(lam_i[i])) < Lp:
                B.append(lam_i[i])
                A.append(self.H[i])
                
        return A,B