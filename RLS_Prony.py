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
        
        # Initialise pseudo inverses
        self.P1 = np.matrix(np.identity(self.M))
        self.P2 = 10e100*np.matrix(np.identity(self.M))
        
        # Initialise errors
        self.e1 = 0
        self.e2 = 0
        
        self.A = np.matrix(np.zeros([M,1]))
        self.r = np.array(np.zeros([M,1]))
        self.H = np.matrix(np.zeros([M,1]))  
        
        self.obs_buffer = [0]
        
        # Fill with buffer with zeros up to the model order
        while(len(self.obs_buffer) < (self.M)):
            self.obs_buffer.insert(0,0)  
        
        self.num_obs = 0
        
        self.y_est = 0
        
    def update(self,y):
        # Test SSE with current model order
        
        # Increase model order by one if enough error
        
        # Update
        self.lpm(y)
 
        # Slide the input window along
        self.obs_buffer.insert(0,-y)
        self.obs_buffer.pop()
     
        self.num_obs += 1
        
        # if self.num_obs > self.M:         
        self.prony()
        
        
    def lpm(self,y):
        lam_inv = 1/self.lam
        
        # Get past inputs [y(N-2) y(N-3) ... y(N-M)]
        un = np.asmatrix(self.obs_buffer).T 
        
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
        
        #self.y_est = np.asscalar(self.A.T@un)
   
    def prony(self):
        lam_inv = 1/self.lam
        y = -np.array(self.obs_buffer)
        
        self.r = self.get_roots()
            
        # Create new row of the Vandermonde matrix
        uz = np.asmatrix(np.zeros([self.M,len(self.r)],dtype=complex))
        
        for i in range(self.M):
            for j in range(len(self.r)):
                uz[i,j] = self.r[j]**(i)
            
        # Replace Inf values with max     
        uz = self.remove_inf(uz)
        
        #self.H = np.array(np.linalg.lstsq(uz,y)[0])
        
        self.H = np.array(lstsq(uz,y)[0])
        
        # # Iterate algorithm
        # #Pn_z = self.P2.copy()
        # Pn_z = 100*np.matrix(np.identity(self.M))
        
        # K = (lam_inv*Pn_z@uz)/(1+lam_inv*uz.T@Pn_z@uz) # Gain
        # K = self.remove_inf(K)
    
        # self.e2 = np.asscalar(y - self.H.T@uz) # Test to see how the old estimate compares to the new data
    
        # self.H = K*self.e2
    
        # #self.H = self.H + K*self.e2 # New coefficients are the old + the gain * the error
    
        # self.P2 = lam_inv*Pn_z - lam_inv*K@uz.T@Pn_z
        
        # self.y_est = np.asscalar(self.H.T@uz)
        
        return
    
    def get_prony(self):
        pA = self.H
        
        rts = self.get_roots()     
        Lp = math.inf
        
        # Remove non-dominant poles
        if np.count_nonzero(rts) > 1:
            # d = rts[np.real(rts) < 0]
            # d = np.min(-np.real(d))
            #d = np.min(np.abs(np.real(rts)))
            d = np.min(rts[np.real(rts) >= 0])
            Lp = 10*np.real(d) # Set poles threshold
        
        #print(Lp)
        
        A = []
        B = []
        
        for i in range(len(pA)):           
            if np.real(np.abs(np.real(rts[i]))) <= Lp:

                B.append(rts[i])
                A.append(pA[i])
                
        B = np.log(np.abs(B))/self.Ts
        B[np.isneginf(np.real(B))] = 0
        return A,B
    
    def get_est(self,t): 
        
        y_est = 0
        k = (self.num_obs)*self.Ts
        k = 0
        
        A,B = self.get_prony()
        
        if len(A) != len(B):
            print('zero')
            return 0
             
        for j in range(len(A)):
            y_est = y_est + np.asscalar(A[j]*np.exp(B[j]*k))
        return y_est
        

    def get_error(self):
        '''
        Finds the a priori (instantaneous) error. 
        Does not calculate the cumulative effect
        of round-off errors.
        '''
        return self.e1
    
    def get_roots(self):
        # Find the roots of the characteristic equation
        c = np.asarray(self.A).reshape(-1)
        c = np.insert(c,0,1)
        r = poly.polyroots(c)
        
        while (self.M > len(r)):
            r = np.append(r,0)
        
        return r

    
    def predict(self, x):
        '''
        Predict the value of observation x. x should be a numpy matrix (col vector)
        '''
        return float(self.w.T*x)
    
    def remove_inf(self,mat):
        mat_r = np.asarray(np.real(mat))
        mat_i = np.asarray(np.imag(mat))
        
        mat_r[np.isinf(mat_r)] = sys.float_info.max*np.sign(mat_r[np.isinf(mat_r)])
        mat_i[np.isinf(mat_i)] = sys.float_info.max*np.sign(mat_i[np.isinf(mat_i)])
        
        return mat_r + 1j*mat_i
    
    def get_dom_poles(self,r):
        
        if not np.any(r):
            return r
        
        # Remove non-dominant poles
        d = r[np.real(r) < 0]
        d = np.min(-np.real(d))
        Lp = 5*d # Set poles threshold
        
        return r[np.abs(np.real(r)) <= Lp]