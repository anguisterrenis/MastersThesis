# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 15:17:39 2015

@author: Daniel Rose

Definition of an Ornstein-Uhlenbeck process as subclass of a general stochastic 
process.
"""

import numba as nb
import numpy as np

from stochastic_process import StochasticProcess, StochasticProcess_with_PSD_Iteration
from autoregressive import AutoregressiveStochasticProcess
from parallel_ff_network import FeedForwardNetwork


@nb.jit        
def ornstein_uhlenbeck(k, dt, v0, D, signal):
    '''Function that defines an Ornstein-Uhlenbeck process with a potential
    U = x^2/2 and an equation of motion x' = -U'(x) + f(t).
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
        v += dt*(-v + sigma*signal[i])
        traj[i] = v
    return traj

def func_OUP_PSD_gamma(f, D, gamma):
    return (D/(gamma**2 + (2*np.pi*f)**2))
    
def func_OUP_PSD_tau(f, D, tau):
    w = 2*np.pi*f
    return (D/(1 + (w*tau)**2 ))
    
def func_OUP_PSDn_coeff(n):
    if n==0:
        return 1 # should be dt, but this is simpler. 
    else:
        coeff = 2.
        for i in range(1,n):
            coeff *= (2*i)/(2*i-1)
        return coeff
        
        
def func_OUP_PSDn_norm(f, n):
    return func_OUP_PSDn_coeff(n)* 1/(1+(2*np.pi*f)**2)**n

def func_OUP_PSDn(f, n, D2):
    return func_OUP_PSDn_norm(f, n-1)*D2/(1+(2*np.pi*f)**2)
    
def func_OUP_PSDn_non_norm(f, n, D2):
    return (D2/(1+(2*np.pi*f)**2))**n
    
@nb.jit()
def gen_OUP_Zn_non_norm(nmax, D, dt):
    coeff = 1./dt
    yield D/(2*dt)
    for i in np.arange(1,nmax): #stops at n-1
        coeff *= 1 - 1/(2*i)
        yield coeff * D**(i+1)/2
        
def func_OUP_v0_PSD(f, sigma_v0, T): 
    return sigma_v0**2/(1+(2*np.pi*f)**2)/T
    
def func_OUP_v0_PSD_tau(f, sigma_v0, T, tau): 
    return (sigma_v0*tau)**2/(1+(2*np.pi*f*tau)**2)/T
    
def func_OUP_PSDn_finite_T(f, n, T, tau, v0_sigma):
    '''original function for the PSD of the nth generation OUP + correction 
    term in first approximation'''
    PSD = (func_OUP_PSDn_norm(f, n)
           + (tau*v0_sigma)**2/(1+(2*np.pi*f*tau)**2)/T
           )
    return PSD
    
def func_OUP_PSDn_finite_T_vT(f, n, T, tau, v0_sigma,vT_sigma):
    '''original function for the PSD of the nth generation OUP + correction 
    term in first approximation'''
    PSD = func_OUP_PSDn_finite_T(f, n, T, tau, v0_sigma) + (vT_sigma)**2/(1+(2*np.pi*f*tau)**2)/T
    return PSD
    
def func_OUP_PSD2_FTW(f,T,D):
    '''Power spectrum of 2nd generation with FTW correction
    D = sigma^2'''
    w = 2*np.pi*f    
    
    PSD = D**2*(1/(1+w*w)**2 + 1/(2*T*(1+w*w)))
    return PSD
    
def func_OUP_PSDn_FTW(f,n,T,D):
    '''Power spectrum of 2nd generation with FTW correction
    D = sigma^2'''
    w = 2*np.pi*f    
    Zn = 1
    for i in range(1,n): # stops at n-1
        Zn *= 1 - 1/(2*i)
    PSD = D**n*(1/(1+w*w)**n + Zn/(T*(1+w*w)))
    return PSD
    


class OUProcess(StochasticProcess):
    '''Class that defines a stochastic process in a square potential with
    equation U(x) = x*x/2
    
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
        
        
        StochasticProcess.__init__(self, ornstein_uhlenbeck, *args, **kwargs)
        
        

class IteratedOUProcess(StochasticProcess_with_PSD_Iteration):    
    '''Class that defines a stochastic process in a square potential with
    equation U(x) = x*x/2.
    
    Input attributes:
    * k         : number of iterations
    * dt        : time step
    * v0        : initial condition
    * D         : signal intensity (rescaling of signal)
    * signal_PSD: (optional) external signal
    * N         : (optional) number of realization for computation of PSD
    '''
    
    def __init__(self, *args, **kwargs):
        '''Initializing the general process class with the OU process as 
        'process' method.'''
        
        super().__init__(ornstein_uhlenbeck, *args, **kwargs)
        
class AutoregressiveOUProcess(AutoregressiveStochasticProcess):
    '''Class that defines a stochastic process in a square potential with
    equation U(x) = x*x/2.
    
    Input attributes:
    * k         : number of iterations
    * dt        : time step
    * v0        : initial condition
    * D         : signal intensity (rescaling of signal)
    * signal_PSD: (optional) external signal
    * NReal     : (optional) number of realization for computation of PSD
    '''
    def __init__(self, *args, **kwargs):
        '''Initializing the general process class with the OU process as 
        'process' method.'''
        
        super().__init__(ornstein_uhlenbeck, *args, **kwargs)
        
        
###############################################################################
############### Feed-Forward network version ##################################
###############################################################################

@nb.jit(nopython=True)
def linear_process_FFN(trajArr,v0Arr,dt,D):
    '''Function defining a parallized computational feed-forward network 
    of linear elements.
    Input:
    * trajArr  : array of trajectories from previous generation
                  assumed to be normalized w.r.t. variance
    * v0Arr     : array of initial conditions for every trajectory
    * dt        : time step
    * sigma     : standard deviation or scaling factor for input signal
    Output:
    * trajArr   : new trajectories (old are supposed to be overwritten)
    '''
    
    # shorthands
    k = trajArr.shape[1]
    
    sigma = np.sqrt(D)
    
    # compute first value from initial condition 
    # (actually skipping the IC in the output)
    trajArr[:,0] = (1-dt) * v0Arr[:] + sigma*dt*trajArr[:,0]
    
    # now do the main loop over the rest of the trajectories
    for i in range(1,k):
        trajArr[:,i] = (1-dt) * trajArr[:,i-1] + sigma*dt*trajArr[:,i]
        
    return trajArr
    
class FFN_OUP(FeedForwardNetwork):
    '''Subclass of feed-forward networks for ornstein-uhlenbeck process/linear elements
    Input attributes:
    * k         : number of iterations
    * dt        : time step
    * v0        : initial condition
    * sigma     : signal intensity (rescaling of signal)
    * signal_PSD: (optional) external signal
    * NReal         : (optional) number of realization for computation of PSD
    '''
    
    #required_keys = ['k', 'dt', 'v0', 'D','NReal']
    
    def __init__(self, *args, **kwargs):
        '''Initializing the general process class with the OU process as 
        'process' method.'''
        
        super().__init__(linear_process_FFN,ornstein_uhlenbeck, *args, **kwargs)
