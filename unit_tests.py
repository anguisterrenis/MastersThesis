# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 11:58:41 2016

Functions defined for unit testing for various scenarios.

@author: Daniel Rose
"""

import numba as nb
import numpy as np

from stochastic_process import StochasticProcess_with_PSD_Iteration

def standard_params(dt=0.1,D=1, v0=0., normalize=False):
    '''yield a dictionary of predefined standard parameters'''
    params = {}
    T = 1000
    k = int(T/dt)
    NReal = 1000
    NIter = 10
    params['dt'] = dt
    params['k'] = k
    params['v0'] = v0
    params['D'] = D
    params['NReal'] = NReal
    params['normalize'] = normalize
    
    return NIter, params
    

###############################################################################
################### Iterated Stochastic Process class tests ###################
###############################################################################

################### Normalisation #############################################


@nb.jit        
def rescale_process(k, dt, v0, D, signal):
    '''Process that simply rescales the incoming signal by D'''
    traj = signal*np.sqrt(D)
    return traj
    
class IteratedRescaleProcess(StochasticProcess_with_PSD_Iteration):
    '''Class that defines a process for unit testing that simply rescales the 
    incoming signal. This is meant to be used to test the normalization 
    procedures implemented in the parent class'''
    
    def __init__(self, *args, **kwargs):
        '''Initializing by calling the 'init' method of the super class.'''        
        
        super().__init__(rescale_process, *args, **kwargs)
        # also? super().__init__(self, bistable_process, *args, **kwargs)


################### Average subtraction in PSD ################################

@nb.jit
def gaussian_white_noise(k, dt, v0, D, signal):
    '''Process that simply gives 'k' gaussian numbers, centered at 'v0' 
    and with variance 'D'.
    '''
    sigma = np.sqrt(D)
    traj = np.random.normal(loc=v0,scale=sigma,size=k)
    return traj
    
class GaussianWhiteNoise(StochasticProcess_with_PSD_Iteration):
    '''Class that defines a process for unit testing that simply gives 'k' 
    gaussian numbers, centered at 'v0' and with variance 'D', ignoring the 
    signal. This is meant to be used to test the average subtraction in the 
    PSD calculation.'''
    
    def __init__(self, *args, **kwargs):
        '''Initializing by calling the 'init' method of the super class.'''        
        
        super().__init__(gaussian_white_noise, *args, **kwargs)
        # also? super().__init__(self, bistable_process, *args, **kwargs)
        

######################### Network process #####################################
