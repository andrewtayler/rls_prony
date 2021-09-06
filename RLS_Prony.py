#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 13:46:03 2021

@author: andrew
"""

import numpy as np
import numpy.polynomial.polynomial as poly
import sys

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
        self.P2 = np.matrix(np.identity(self.M))
        
        # Initialise errors
        self.e1 = 0
        self.e2 = 0
        
        self.A = np.matrix(np.zeros([M,1]))
        self.r = np.array(np.zeros([M,1]))
        self.H = np.matrix(np.zeros([M,1]))  
        
        self.obs_buffer = []#np.matrix(np.zeros([1,M]))
        self.obs_b2 = []
        
        self.num_obs = 0
        
        self.y_est = 0
        
    def updateLPM(self,y):
        lam_inv = 1/self.lam
        
        # Fill with buffer with the first few values
        if(len(self.obs_buffer) < (self.M)):
            self.obs_buffer.insert(0,-y)            
        else:
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
            
            # Slide the input window along
            self.obs_buffer.insert(0,-y)
            self.obs_buffer.pop()
     
        self.num_obs += 1
        
    def updateProny(self,i):
        lam_inv = 1/self.lam
        y = -self.obs_buffer[0]
        
        # Find the roots of the characteristic equation
        c = np.asarray(self.A).reshape(-1)
        c = np.insert(c,0,1)
        self.r = poly.polyroots(c)
        
        
        # # Remove non-dominant poles
        # d = r[np.real(r) < 0]
        # d = np.min(-np.real(d))
        # Lp = 5*d # Set poles threshold
        
        # self.r = r[np.abs(np.real(r)) <= Lp]
        # idx = np.where(np.abs(np.real(r)) <= Lp)
        
        # if len(self.P2) != len(self.r):
        #     self.P2 = np.matrix(np.identity(len(self.r)))
        #     self.H = np.matrix(np.zeros([len(self.r),1]))
        #     print(f'Num roots changed to {len(self.r)} at {i}')
            
        # Create new row of the Vandermonde matrix
        uz = np.asmatrix(np.zeros([len(self.r),1],dtype=np.clongdouble))
        for j in range(len(self.r)):
            uz[j,:] = self.r[j]**(self.num_obs-1)
        
        # Replace Inf values with max     
        uz = self.remove_inf(uz)
        
        # Iterate algorithm
        Pn_z = self.P2
        
        K = (lam_inv*Pn_z@uz)/(1+lam_inv*uz.T@Pn_z@uz) # Gain
        K = self.remove_inf(K)
    
        self.e2 = np.asscalar(y - self.H.T@uz) # Test to see how the old estimate compares to the new data
    
        self.H = self.H + K*self.e2 # New coefficients are the old + the gain * the error
    
        self.P2 = lam_inv*Pn_z - lam_inv*K@uz.T@Pn_z
        
        self.y_est = np.asscalar(self.H.T@uz)
        
        return

    def get_error(self):
        '''
        Finds the a priori (instantaneous) error. 
        Does not calculate the cumulative effect
        of round-off errors.
        '''
        return self.e2
    
    def get_roots(self):
        return self.r
    
    def get_prony(self):
        pA = self.H
        pB = np.log(self.get_roots())/self.Ts
        return pA,pB
    
    def get_est(self,t): 
        y_est = 0
        # for A,B in enumerate(zip(*self.get_prony())):
        #     y_est = y_est + A*np.exp(B*t)
        A,B = self.get_prony()
             
        for j in range(len(A)):
            y_est = y_est + np.asscalar(A[j]*np.exp(B[j]*t))
        return y_est
    
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
    
    def get_dom_poles(self):
        r = self.r
        # Remove non-dominant poles
        d = r[np.real(r) < 0]
        d = np.min(-np.real(d))
        Lp = 5*d # Set poles threshold
        
        return r[np.abs(np.real(r)) <= Lp]