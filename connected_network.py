# -*- coding: utf-8 -*-
"""
Created on Wed May  4 10:03:22 2016

Project for sparsely connected network with random, sparse connections.

@author: Daniel Rose
"""

import numpy as np
#import numba as nb

from stochastic_process import Parameters, InitializationError 
from stochastic_process import StochasticProcess

class NetworkProcess(StochasticProcess):
    '''Class for a network of equal elements
    Required parameters:
    * N         : Network size
    * k         : number of points of one trajectory (iterations)
                    usage: k = T/dt
    * dt        : time step (for the visibile trajectories)
    * v0        : (array containing) initial condition(s)
    * NSamples  : number of samples per element
    * dtSkipped : time step that is calculated but not visible, must be smaller than dt
    '''
    
    required_keys = ['N','k','dt','v0','NSamples','dtSkipped']
    
    def __init__(self, process, coeffsBase=[], samples=[[]], initTraj=True, 
                 countJumps=False, **params):
        '''Need to define 'process' for this to work properly'''
        
        self.check_keys(params)        
        
        self.params = Parameters(**params)      
        
        self.process = process
        
        self.update_coeffs(coeffsBase)
        
        self.update_samples(samples)
        
        v0 = self.params.v0
        self.update_inputs(v0)
        
        self.init_outputs()
        
        dtOutside = self.params.dt
        dtInside = self.params.dtSkipped
        self._dtSkip =  np.int(dtOutside/dtInside)
        
        
        if initTraj:
            self.init_trajectories()
        else:
            self.just_initialized=False
            
        if countJumps:
            self.init_jump_count_and_sate()
        
        
        
    def check_keys(self, params):
        '''Check whether all required keys are given in 'params'
        Input:
        * params : supposed to be a dictionary of parameters
        Output: True or False
        '''
        
        if any((key not in params for key in self.required_keys)):
            raise KeyError("Could not initialize object. "
            "At least one parameter is missing."
            "The following parameters are required: %s" %str(self.required_keys)
            )
            
    def update_coeffs(self, coeffsBase):
        '''Update internal array of coefficients, must be of length 'NSamples' 
        '''
        NSamples = self.params.NSamples      
        N = self.params.N
        
        if len(coeffsBase) == NSamples: #all coeffs given
            coeffs = [coeffsBase]   
            self.coeffs = np.repeat(coeffs, repeats=N, axis=0)
        elif len(coeffsBase) == 0: #no coeffs given
            self.coeffs = np.ones((N,NSamples), dtype=float)
        else:
            raise InitializationError("Length of coefficients array is "
            "inconsistent with number of samples(=%i)" %NSamples)
            
            
    def update_inputs(self, inputs):
        '''Update inputs array'''
        
        N = self.params.N
        
        if len(inputs) == N:
            self.inputs = np.array(inputs,dtype=float)
        elif len(inputs) == 1:
            self.inputs = np.ones(N,dtype=float)*inputs
        else: 
            raise InitializationError("Number of initial conditions is "
            "inconsistent with number of elements(=%i)" %N)
    
    def init_outputs(self):
        '''Initialize internal outputs array'''
        
        N = self.params.N
        
        self.outputs = np.zeros(N,dtype=float)
        
    def update_samples(self, samples=[[]]):
        '''Choose which elements sample over which elements. Outputs an array
        of integer indeces.
        '''
        
        N = self.params.N
        NSamples = self.params.NSamples
        
        if np.any(samples):
            if np.shape(samples) == (N,NSamples):
                self.samples = np.array(samples,dtype=int)
            else: 
                raise InitializationError("Number of samples is "
                "inconsistent with number of elements(=%i) and samples(=%i)." 
                %(N,NSamples))
        else:
            samples = np.zeros((N,NSamples),dtype=int)
            
            for elem in range(N):
                sampleSpace = [x for x in range(N) if x != elem]
                samples[elem,:] = np.random.choice(sampleSpace, NSamples, 
                                                    replace=False)
            
            self.samples = samples
        
        
    def init_trajectories(self):
        '''Initialize internal trajectories array'''

        N = self.params.N # network size = axis 0
        k = self.params.k # number of stored points in time = axis 1
        
        try:
            self.trajectories.fill(0.)
            print('Filled existing trajectories with zeros')
        except AttributeError:
            self.trajectories = np.zeros((N,k),dtype=float)*1. #really create that array
            print("Created new zero-valued trajectories")
        self.trajectories[:,0] = self.inputs[:] #use IC as first point
        self.just_initialized=True
        
    def init_jump_count_and_sate(self):
        '''initialize internal jump state array (where are we now?) and jump
        count.'''
        
        N = self.params.N
        self.jumpStates = np.ones(N,dtype=int)
        self.jumpStates[:] *= np.sign(self.inputs).astype(int)
        # assume, that all inputs are different from zero, 
        # otherwise it doesn't work
        if np.any(self.jumpStates[:] == 0):
            raise InitializationError("Some of the jump states are 0. " 
            "That doesn't make sense - or at least it didn't in the past.")
            
        self.jumpCount = int(0)
        # replace original compute function with modified version
        self.compute_trajectories = self.compute_trajectories_and_count_jumps
        print("Jumps are counted in trajectories.")
        
    def get_trajectories(self,continued=False):
        '''internal trajectory array needs to initialized for this to work. 
        might be better to put outside class with @numba.jit decorator.'''
        
        if continued:
            self.update_inputs(self.trajectories[:,-1])
            self.init_outputs()
            self.init_trajectories()
            self.compute_trajectories()
        elif self.just_initialized:
            self.compute_trajectories()
        else: 
            pass
        
        return self.trajectories
        
    def get_trajectory(self, index=0):
        '''Simple output function to yield a single trajectory instead of the 
        whole array'''
        trajArr = self.get_trajectories()
        traj = trajArr[index]
        return traj
    
    def compute_trajectories(self):
        '''internal trajectory array needs to initialized for this to work. 
        might be better to put outside class with @numba.jit decorator.'''
        

        dtInside = self.params.dtSkipped # note: this is not the dt that is visible from the class
        k = self.params.k
        skip = self._dtSkip
        #N = self.params.N
        
        for i in range(1,k):
            
            self.process(dtInside, self.inputs, self.samples, self.coeffs, skip, self.outputs)
            
            self.trajectories[:,i] = self.outputs[:]
            self.inputs[:] = self.outputs[:]
        
        self.just_initialized=False
        
    def compute_trajectories_and_count_jumps(self):
        '''the above method + jump counts in the process
        does not work, if the wrong process is inserted.'''
        
        dtInside = self.params.dtSkipped # note: this is not the dt that is visible from the class
        k = self.params.k
        skip = self._dtSkip
        #N = self.params.N
        # jump count is reinitiliazed 
        self.jumpCount = 0
        
        for i in range(1,k):
            
            newCount = self.process(dtInside, self.inputs, self.samples, self.coeffs, skip, 
                                    self.jumpStates, self.outputs)
            self.jumpCount += newCount
            self.trajectories[:,i] = self.outputs[:]
            self.inputs[:] = self.outputs[:]
        
        self.just_initialized=False     

        
        
    def get_averages(self):
        '''compute and return the time-dependent ensemble average of the stored
        trajectories'''
        
        trajs = self.get_trajectories()
        avg = np.mean(trajs,axis=0)
        
        return avg
     
    def get_stdevs(self):
        '''compute and return time-dependent ensemble standard deviation of 
        stored trajectories '''
        
        trajs = self.get_trajectories()
        stdev = np.std(trajs,axis=0)
        
        return stdev

    def get_PSD(self,NIter=0,recompute=False,**kwargs):
        '''Compute a power spectrum from multple trajectories
        
        Input:
        * NReal     : number of realizations to compute
        * recompute : False/True. whether or not to recompute the PSD
        
        Output:
        * PSD   : power spectrum
        ''' 
        
        neednewPSD = False        

        if hasattr(self, 'PSD'):
            if recompute:
                neednewPSD = True
        else: 
            neednewPSD = True
            
        if neednewPSD:
            self.compute_PSD(NIter,**kwargs)
        
        return np.array(self.PSD) # copy
        
    def compute_PSD(self,NIter=0,**kwargs):
        '''Compute a power spectrum from multple trajectories
        
        Input:
        * NReal     : number of realizations to compute
        
        Output:
        * PSD   : power spectrum'''
        
        k = self.params.k
        dt = self.params.dt
        NReal = self.params.N        
        
        PSD = np.zeros(int(self.params.k/2+1))  
        
        #outside, in case it hasn't been computed yet
        trajectories = self.get_trajectories() 

        if NIter == 0: # just calculate PSD from known trajectories
            for traj in trajectories:
                ft = self.get_rFFT(traj)
                PSD += abs(ft)**2
                
        else: # repeat calculations
            for i in range(NIter):
                trajectories = self.get_trajectories(continued=True)
                for traj in trajectories:
                    ft = self.get_rFFT(traj)
                    PSD += abs(ft)**2
            
            NReal *= NIter

        
        PSD /= NReal*k*dt
        
        self.PSD = PSD
        
    def get_switching_rate(self):
        '''Simple function: divide jump count by T and N. result:
        jump rate per time unit and per network element.'''
        
        T = self.params.k*self.params.dt
        N = self.params.N
        
        rate = self.jumpCount/T/N
        
        return rate
        
    def get_switching_points(self):
        print("Switching point calculation is not implemented.")
        
    def get_variance_from_PSD(self,*args,**kwargs):
        '''Compute variance of stored PSD by integrating it'''
        
        PSD = self.get_PSD(*args, **kwargs)
        df = self.get_df()
        #variance = df*(PSD[0]+ 2*PSD[1:].sum()) # naive integral
        variance = 2*np.trapz(PSD, dx=df) # trapezoidal integral
        return variance
        
    def get_correlation_time(self,*args,**kwargs):
        '''Compute the correlation time from a stored PSD'''
        
        PSD = self.get_PSD(*args,**kwargs)        
        df = self.get_df()

        PSD_sq_int = 2*np.trapz(PSD*PSD, dx=df)
        PSD_int = 2*np.trapz(PSD, dx=df)
        tau = 2*np.pi * PSD_sq_int/(PSD_int*PSD_int)     

        return tau
        
        
            
            
            
        
        
        
        
        

            
            
            
        