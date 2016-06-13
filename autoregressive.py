# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 21:59:16 2016

@author: Storm

Define stochastic process subclass for autoregressive processes.
"""

from stochastic_process import InitializationError, StochasticProcess_with_PSD_Iteration

import numpy as np
import numba as nb
from scipy.linalg import circulant
    
@nb.jit(nopython=True)
def autoregression(signal, cVec, sigma, k):
    
    depth = len(cVec)
    
    for i in range(k):
        signal[i] = np.random.normal(0,sigma)
        for j in range(depth):
            signal[i] += cVec[j] * signal[i-j-1] 
    return signal

class AutoregressiveStochasticProcess(StochasticProcess_with_PSD_Iteration):
    '''Class that defines a stochastic process simulated by an autoregressive 
    (AR) process of order p (ARdepth).'''
    
    required_keys = StochasticProcess_with_PSD_Iteration.required_keys + ["ARdepth"]
    
    def __init__(self, process, PSDs=[None], normalize=True, **params):
        '''Initializing the general stochastic process plus some extras
        Input:
        * process   : a specific process 
        * PSDs      : previously calculated PSDs
        * normalize : whether or not the output PSD of each generation is 
                      normalized as input for the next generation
                      default: True
        * ARdepth   : (required parameter) depth of the autoregressive process
        '''
        
        super().__init__(process, PSDs, normalize, **params)
        
        if self.params.ARdepth > (self.params.k/2+1):
            raise InitializationError("Autoregression depth 'ARdepth' must be smaller than or equal to half the number of points. Please reinitialize properly.")
        
    # now redefine some of the intialization procedures
    # we keep the update_PSDs procedure
    
    def update_v0Arr(self):
        '''Check integrity of data for initial conditions and transfer them to 
        'v0Arr' array of initial conditions. For the autoregressive process, 
        only one initial condition is necessary - for now.'''
        
        # initialize v0Arr
        
        if np.shape(self.params.v0) == (): # scalar
            self.v0Arr = np.empty((1,1)) # ensure consistency of shape with later implementations
            self.v0Arr[0,0] = self.params.v0
            
        else: # non-scalar (vector or 2D matrix)
        # this part might be useful, if different autoregressive trajectories 
        # need to be started from different initial conditions
        
            if np.shape(np.shape(self.params.v0)) == (2,): # 2D array
                self.v0Arr = np.array(self.params.v0)
                
            elif np.shape(np.shape(self.params.v0)) == (1,): # 1D array
                self.v0Arr = np.empty((1,len(self.params.v0)))
                self.v0Arr[0,:] = self.params.v0[:]
                
            else: 
                raise InitializationError("Dimension of initial conditions (v0) is inconsistent. \
Initialization failed.")

    # redefine the signal production procedure

    def use_as_signal_PSD(self, PSD, normalize=True):
        '''Copy input PSD to self.signal_PSD and prepare autoregressive 
        process.
        Input: 
        * PSD       : Pointer to PSD that is copied to internal signal_PSD
        * normalize : (default: True) whether or not the PSD should be
                      normalized before copying it.
        Output:
        * signal    : a first iteration of the signal is saved as internal
                      array and later used to produce further signals
        '''
        
        # pass PSD to self.signal_PSD, normalize by default
        super().use_as_signal_PSD(PSD, normalize) 
        
        # get correlation function from PSD through irFFT
        # number of used points depends on predefined depth/rank of the 
        # autoregressive process
        print("Preparing autoregressive process...")
        CF = self.get_irFFT(self.signal_PSD, self.params.k)[:self.params.ARdepth]
        
        self.sigma, self.cVec = self.prepare_autoregression_coefficients(CF)
        
        # prepare first random signal
        self.signal = np.random.normal(0,1,self.params.k)
        # save first iteration of signal
        self.signal = self.get_signal()
        

    def get_signal_from_PSD(self):
        '''Compute stochastic signal from given PSD as autoregressive process
        --> first FT to obtain correlation function, then AR
        pretty memory intensive in the current setup.
        
        Output:
        * signal    : signal to be used as input in stochastic process'''
        
        #check if signal PSD exists, remnant from original class  
        try:
            self.signal = autoregression(self.signal, self.cVec, self.sigma,self.params.k)
        except AttributeError:
            print("Signal generation unsufficiently prepared. Run 'use_as_signal_PSD'")
            raise
        else:
            # return signal (not necessary but consistent with earlier implementations,
            # saving some lines of code)
            return self.signal 

    # and now define the autoregression-specific procedures
    def prepare_autoregression_coefficients(self,CF):
        '''Calculate autoregression coefficients (cVec) and variance of 
        additional noise term (sigma**2).
        Is it possible to simplify the matrix inverse in case of a circulant
        matrix?'''
        B = circulant(CF[:-1]) # up to k-1
        Binv = np.linalg.inv(B) # invert matrix
        cVec = np.dot(Binv,CF[1:]) # from 1 to k
        sigma = np.sqrt(CF[0] - np.dot(cVec,CF[1:]))
        
        return sigma, cVec
        
    # redefine the trajectory-getter
#    def get_trajectory(self, v0, signal):
#        '''Collect a trajectory from process, wrapping all constants.
#        Here, a specific initial value (v0) must be given and the signal is 
#        defined externally
#
#        Input:
#        * Signal    : externally defined stochastic signal
#        * v0        : trajectory-specific initial condition. 
#        
#        Output:
#        * traj      : trajectory as numpy array
#        '''
#        traj = self.process(self.params.k, self.params.dt, v0, self.params.D, signal)
#        
#        return traj
        
    # and finally redefine procedures to calculate PSD
        
    def compute_PSD(self, v0Arr):
        '''Compute a power spectrum from multple trajectories.
        Here, the initial condition 'v0' is assumed to be an array 
        (possibly of length 1). This is also a pseudo-parallel structure
        built on top of a serial implementation.
        
        Input:
        * v0Arr     : array containing initial conditions for each trajectory
                        now assumed to be internal variable
        
        Output:
        * PSD       : power spectrum
        * lastValArr: array containing last value of each trajectory (for IC)
        ''' 
        np.random.seed()

        # initialize new PSD and lastValArr
        PSD = np.zeros((1,self.params.k/2 + 1))
        nextv0Arr = np.empty((1,len(v0Arr)))
        nextv0Arr[0,:] = v0Arr[:] # lazy, pass IC to next generation

        # doing this outside the loop to shorten calculation time
        # might be enough to use the same signal for different ICs
        
        for v0 in v0Arr: 
        # start each series at different IC (e.g. valleys of potential)
            # compute initial dummy trajectory to obtain inititial condition
            newv0 = self.get_trajectory(v0)[-1]

            for i in range(self.params.NReal):
                traj = self.get_trajectory(newv0) # use last value as IC
                newv0 = traj[-1]
                ft = self.get_rFFT(traj)
                PSD += (ft*ft.conjugate()).real # ignoring the +0j
            
        # normalize (*1/(N*T))
        PSD /= self.params.NReal*self.params.k*self.params.dt
        
        return PSD, nextv0Arr
        
    # and now add an additional term to the read/write procedures
        
    def write_PSDs_to_file(self, prefix, folder='', suffix='', **kwargs):
        '''Write computed PSDs to file as well as the v0Arr'''
        
        newSuffix = "_ARdepth%i%s" %(self.params.ARdepth,suffix)
        super().write_PSDs_to_file(prefix, folder, suffix=newSuffix, **kwargs)
        
    
    def read_PSDs_from_file(self,NIter,prefix,folder='',suffix='', **kwargs):
        '''Read PSD and v0Arr data from file and save data
        to instance'''
        
        newSuffix = "_ARdepth%i%s" %(self.params.ARdepth,suffix)
        super().read_PSDs_from_file(NIter, prefix, folder, suffix=newSuffix, **kwargs)
    
    