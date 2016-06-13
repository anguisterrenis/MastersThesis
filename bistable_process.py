# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 11:20:33 2015

@author: Daniel Rose

Definition of classes for stochastic processes. Let's try it.

"""

import numba as nb
import numpy as np

from stochastic_process import StochasticProcess, StochasticProcess_with_PSD_Iteration


@nb.jit        
def bistable_process(k, dt, v0, D, signal):
    '''Function that defines a bistable process with a potential
    U = x^4/4 - x^2/2 and an equation of motion x' = U'(x) + f(t).
    Input: 
    * k      : number of points
    * dt     : time step
    * v0     : initial condition
    * D      : scaling factor for external signal
    * signal : externally defined signal
    Output:
    * traj   : full trajectory as numpy array
    '''
    
    sigma = np.sqrt(D)
    v = v0
    traj = np.empty(k)
    for i in np.arange(k):
        v += dt*(-v**3 + v + sigma*signal[i])
        traj[i] = v
    return traj
    

class BistableProcess(StochasticProcess):
    '''Class that defines a stochastic process in a bistable potential with
    equation U(x) = x**4/4 - x*x/2
    
    Input attributes:
    * k         : number of iterations
    * dt        : time step
    * v0        : initial condition
    * D         : signal intensity (rescaling of signal)
    * signal_PSD: (optional) external signal
    * N         : (optional) number of realization for computation of PSD
    '''

    required_keys = ('k', 'dt', 'v0', 'D')

    def __init__(self, *args, **kwargs):
        '''Initializing the bistable process class by calling the 'init' method
        of the super class.'''        
        
        
        StochasticProcess.__init__(self, bistable_process, *args, **kwargs)
        # also? super().__init__(self, bistable_process, *args, **kwargs)
        
class IteratedBistableProcess(StochasticProcess_with_PSD_Iteration):
    '''Class that defines a stochastic process in a bistable potential with
    equation U(x) = x**4/4 - x*x/2
    
    Input attributes:
    * k         : number of iterations
    * dt        : time step
    * v0        : initial condition
    * D         : signal intensity (rescaling of signal)
    * signal_PSD: (optional) external signal
    * N         : (optional) number of realization for computation of PSD
    '''

    #required_keys = ('k', 'dt', 'v0', 'D')

    def __init__(self, *args, **kwargs):
        '''Initializing the bistable process class by calling the 'init' method
        of the super class.'''        
        
        super().__init__(bistable_process, *args, **kwargs)
        
            
###############################################################################
################ Weak noise approx: Shifted linear process ####################
###############################################################################
   
def func_shifted_lin_PSD(f, n, D2):
    return (D2/(4+(2*np.pi*f)**2))**n
         
@nb.jit
def shifted_linear_process(k, dt, v0, D, signal):
    '''Function that defines a linear process, centered at -1 corresponding to 
    the potential U = (x+1)^2 -1/4 and an equation of motion x' = -U'(x) + f(t).
    Input: 
    * k      : number of points
    * dt     : time step
    * v0     : initial condition
    * D      : scaling factor for external signal (sigma**2)
    * signal : externally defined signal
    Output:
    * traj   : full trajectory as numpy array
    '''
    
    sigma = np.sqrt(D)
    v = v0
    traj = np.empty(k)
    for i in np.arange(k):
        v += dt*(-2*(v + 1) + sigma*signal[i])
        traj[i] = v
    return traj
    
    
class IteratedShiftedLinearProcess(StochasticProcess_with_PSD_Iteration):
    '''Class that defines a stochastic process in a shifted linear potential
    with equation U = (x+1)^2 -1/4
    
    Input attributes:
    * k         : number of iterations
    * dt        : time step
    * v0        : initial condition
    * D         : signal intensity (rescaling of signal)
    * signal_PSD: (optional) external signal
    * N         : (optional) number of realization for computation of PSD
    '''

    #required_keys = ('k', 'dt', 'v0', 'D')

    def __init__(self, *args, **kwargs):
        '''Initializing the class by calling the 'init' method
        of the super class.'''        
        
        super().__init__(shifted_linear_process, *args, **kwargs)
        # also? super().__init__(self, bistable_process, *args, **kwargs)
        
        
###############################################################################
################ Strong noise approx: Plain x^4 potential #####################
###############################################################################
        
@nb.jit        
def plain_x4_process(k, dt, v0, D, signal):
    '''Function that defines a non-linear process with a potential
    U = x^4/4 and an equation of motion x' = U'(x) + f(t).
    Input: 
    * k      : number of points
    * dt     : time step
    * v0     : initial condition
    * D      : scaling factor for external signal
    * signal : externally defined signal
    Output:
    * traj   : full trajectory as numpy array
    '''
    
    sigma = np.sqrt(D)
    v = v0
    traj = np.empty(k)
    for i in np.arange(k):
        v += dt*(-v**3 + sigma*signal[i])
        traj[i] = v
    return traj

        
class IteratedPlainX4Process(StochasticProcess_with_PSD_Iteration):
    '''Class that defines a stochastic process in a shifted linear potential
    with equation U = (x+1)^2 -1/4
    
    Input attributes:
    * k         : number of iterations
    * dt        : time step
    * v0        : initial condition
    * D         : signal intensity (rescaling of signal)
    * signal_PSD: (optional) external signal
    * N         : (optional) number of realization for computation of PSD
    '''

    #required_keys = ('k', 'dt', 'v0', 'D')

    def __init__(self, *args, **kwargs):
        '''Initializing the class by calling the 'init' method
        of the super class.'''        
        
        super().__init__(plain_x4_process, *args, **kwargs)
        # also? super().__init__(self, bistable_process, *args, **kwargs)
        
###############################################################################
############### network class and core process ################################
###############################################################################
     
from connected_network import NetworkProcess     
     
@nb.jit(nopython=True)      
def bistable_network_process(dt, inputs, samples, coeffs, skip, outputs):
    '''Process that calculates one step at a time in a network of bistable 
    elements
    Input:
    * dt     : time step
    * inputs : array of points of the previous iteration step
    * samples: array telling, which network element samples over which other 
                elements, must be 'int'-type array
    * coeffs : ... and with what to multiply those (is N*d array, 
                although d would be sufficient)
    * skip   : number of iterations to compute before returning result
    * outputs: array to output to
    Output:
    * outputs: may not need to be done explicitly - pointers are used to save 
                memory and setup time
    Remark: 
    * no jump rate calculation implemented
    * assumes that first dimension of all arrays is the same, won't work otherwise
    * second dimension of 'samples' and 'coeffs' must be at least 2
    * could be easily transformed into parallel version, by removing the first for-loop
    '''
    
    for iterIndex in range(skip):
        for mainIndex, oldVal in enumerate(inputs):
            newVal = oldVal - oldVal*oldVal*oldVal #the dt comes later
            for coeffIndex, sampleIndex in enumerate(samples[mainIndex]):
                newVal += coeffs[mainIndex,coeffIndex]*inputs[sampleIndex]
            newVal *= dt #this makes it the derivative
            newVal += oldVal #and now add that to the old value
            outputs[mainIndex] = newVal
        inputs[:] = outputs[:]
    
    return True
    
@nb.jit(nopython=True)      
def bistable_network_process_with_jump_count(dt, inputs, samples, coeffs, skip, 
                                             jumpStates, outputs):
    '''Process that calculates one step at a time in a network of bistable 
    elements (and now also counts jumps)
    Input:
    * dt     : time step
    * inputs : array of points of the previous iteration step
    * samples: array telling, which network element samples over which other 
                elements, must be 'int'-type array
    * coeffs : ... and with what to multiply those (is N*d array, 
                although d would be sufficient)
    * skip   : number of iterations to compute before returning result
    * jumpstates : defines current minima (needs to be initialized)
    * outputs: array to output to
    Output:
    * outputs: may not need to be done explicitly - pointers are used to save 
                memory and setup time
    * jumpCount
    Remark: 
    * no jump rate calculation implemented
    * assumes that first dimension of all arrays is the same, won't work otherwise
    * second dimension of 'samples' and 'coeffs' must be at least 2
    * could be easily transformed into parallel version, by removing the first for-loop
    '''
    
    lower = -0.5
    upper = 0.5
    threshold = [0,lower,upper]    
    jumpCount = 0
    
    for iterIndex in range(skip):
        for mainIndex, oldVal in enumerate(inputs):
            # main loop
            newVal = oldVal - oldVal*oldVal*oldVal #the dt comes later
            for coeffIndex, sampleIndex in enumerate(samples[mainIndex]):
                newVal += coeffs[mainIndex,coeffIndex]*inputs[sampleIndex]
            newVal *= dt #this makes it the derivative
            newVal += oldVal #and now add that to the old value
            outputs[mainIndex] = newVal
            
            # jump count loop, could be vectorized (after data transfer)
            # but not necessary for now
            pos = jumpStates[mainIndex]
            if (newVal*pos <= threshold[pos]*pos):
                    jumpCount += 1
                    jumpStates[mainIndex] = (-1)*pos
            
                    
        # data transfer
        inputs[:] = outputs[:]    

    return jumpCount
        
    
class BistableNetworkProcess(NetworkProcess):
    '''Class for network computation of bistable elements'''
    
    def __init__(self,*args,countJumps=False,**kwargs):
        
        if countJumps:
            process = bistable_network_process_with_jump_count
        else:
            process = bistable_network_process
        
        super().__init__(process,*args,countJumps=countJumps,**kwargs)
            
    