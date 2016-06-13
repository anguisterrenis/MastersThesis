# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 16:35:21 2016

Class definition for stochastic class of parallel feed-forward network

@author: Daniel Rose
"""

from stochastic_process import StochasticProcess_with_PSD_Iteration, InitializationError

#import numba as nb
import numpy as np

class FeedForwardNetwork(StochasticProcess_with_PSD_Iteration):
    '''Definition of a class for parallelized feed-forward networks. The core
    parallelized process definition needs to be given externally'''
    
    required_keys = StochasticProcess_with_PSD_Iteration.required_keys + ["D"]
    
    def __init__(self,processFFN,process1D, trajectories=[None], PSDs=[None], normalize=True,
                 skip = 0, signalInit='gw',**params):
        '''Initialize class by first calling the superclass's init method.
        The function "process" is defined differently in this environment, which
        needs to be taken care of in specialized code, overwriting some key functions.
        Input:
        * processFFN    : process defined for the feed-forward network definition'''
        
        super().__init__(process1D, PSDs, normalize,
                 skip, signalInit, **params)
                 
        self.processFFN = processFFN
                 
        self.initialize_trajectories(signalInit, trajectories)
    
    
    def initialize_trajectories(self,signalInit='gw',trajectories=[None]):
        '''Initialize the initial generation of trajectories. The purpose of this
        metod ist that it can be expanded to implement other kinds of initial signals.
        Input:
        * signalInit    : value corresponds to different types of signals
            - 'gw'      : gaussian white noise
        * trajectories  : input trajectories if provided
        Output:
        * self.trajectories : initial trajectories are saved to internal variable
        '''
        
        k = self.params.k
        N = self.params.NReal        

        if np.any(trajectories):
            if np.shape(trajectories) == (N,k):
                self.trajectories = trajectories
            else: 
                raise InitializationError("Shape of input trajectories incompatible with parameters. The shape is %s and should be %s" %(str(np.shape(trajectories)),str((N,k))))
        
        elif signalInit == 'gw':
            self.trajectories = self.get_signal_gaussian_white()
        else: 
            self.trajectories = np.zeroes((N,k)) # initialize as zeros, since it needs to be there
        #make sure you always save the trajectories... otherwise more code is necessary
            
    
        
    def get_signal_gaussian_white(self):
        '''For initialization: "Zeroth" generation of trajectories
        is initialized as Gaussian white noise. This function simply outputs the 
        trajectories as an array'''
        
        k = self.params.k
        N = self.params.NReal 
        dt = self.params.dt
        
        trajArr = np.random.normal(0,1/np.sqrt(dt),(N,k)) # normalize so no specific code is needed in core
        
        return trajArr
        
    # the method 'get_PSD' is taken from the original class
    
    def compute_next_PSD_iteration(self, v0Arr,repeat=True,suppress_f0=False,**kwargs):
        '''Compute a new iteration for the power spectrum.
        Input:
        * v0Arr     : Array containing initial conditions for the next generation
                      here: simply the same for all generations
        * repeat    : (default:True) whether or not to repeat the trajectory simulation
                      to hopefully get stationary distributions
        ''' 
        
        # keep this from the original to ensure consistence with single-trajectory methods
        self.use_as_signal_PSD(self.PSDs[-1,:],self.params.normalize,suppress_f0, **kwargs)
        
        # for now, a single set of initial conditions is used for all generations
        # this may be changed later
        PSD = self.compute_PSD(v0Arr,repeat)
        
        # save results to internal arrays
        self.v0Arr = np.concatenate((self.v0Arr,[v0Arr]), axis=0) # copy of original array, preliminary
        self.PSDs = np.concatenate((self.PSDs,PSD), axis=0)

    
    def compute_PSD(self, v0Arr,repeat=True):
        '''Compute a power spectrum from multple trajectories.
        Here, the initial condition 'v0' is assumed to be an array.
        Implementation for a parallel simulation scheme (vector operations)
        
        Input:
        * v0Arr     : array containing initial conditions for each trajectory
                      (assumed to be internal variable)
        * repeat    : (default:True) whether or not to repeat the trajectory simulation
                      to hopefully get stationary distributions
        
        Output:
        * PSD   : power spectrum
        * lastValArr : array containing last value of each trajectory (for IC)
        ''' 
        np.random.seed() # might make any difference at all... will be moved to process, maybe

        # initialize new PSD
        PSD = np.zeros((1,int(self.params.k/2 + 1)))
        
        # calculate next gen trajectories (are automatically saved to internal array (memory expensive))
        self.get_trajectories(v0Arr,repeat)
        
        # calculate collective rfft
        trajArr = self.trajectories
        fts = self.get_rFFT(trajArr[:,self.params.skip:],axis=-1) 
        # skip first few points, if desired 
        # (might not be stable, not tested, not used)
        
        # calculate PSD as <|ft|^2>/T
        fts = np.abs(fts)**2
        PSD[0,:] = fts.mean(axis=0) # average over ensemble
            
        # normalize with time window T = k*dt
        PSD /= self.params.k*self.params.dt
        
        return PSD
        
        
    def get_trajectories(self, v0Arr, repeat=True):
        '''Calculate new trajectories in parallel and save them to the internal
        'trajectories' array. Should generally not be used outside the 'get_PSD'
        method.
        Input: 
        * v0Arr     : array containing initial conditions for each trajectory
        * repeat    : (default:True) whether or not to repeat the trajectory simulation
                      to hopefully get stationary distributions
        '''
        
        dt = self.params.dt
        D = self.params.D
        
        # dimensions of trajectories:
        # 0 : number of different realizations
        # 1 : single trajectory of length k = (T/dt)
        if repeat: # redo calculations to obtain stationary initial conditions
            # need to copy it though...
            dummyArr = np.array(self.trajectories)
            self.processFFN(dummyArr,v0Arr,dt,D)
            newV0Arr = dummyArr[:,-1]
            self.testVals = newV0Arr
            #newV0Arr = self.processFFN(self.trajectories,v0Arr,dt,D)[:,-1]
            self.processFFN(self.trajectories,newV0Arr,dt,D)
        else: # or don't
            self.processFFN(self.trajectories,v0Arr,dt,D)
            
        
        if self.params.normalize:
            std = self.trajectories[:,-1].std() # calculate standard deviation 
            self.trajectories = self.trajectories/std # normalize variance
            # this could also be achieved by calcualting the variance of the PSD 
            # should yield the same.   
        else: pass
            
        
            
        
            
    
    
    def write_PSDs_to_file(self, prefix, folder='', suffix=''):
        '''Write computed PSDs to file as well as the v0Arr and last 
        trajectories. Writing the trajectories consumes lots of both space
        and time (dependent on data transfer rates). Might not be worth it.'''
        
        import csv
        import os
        
        super().write_PSDs_to_file(prefix,folder,suffix)
        
        trajFilename = "%s_D%f_k%i_dt%.3f_NReal%i_NIter%i_trajArr%s.csv" %(
        prefix, self.params.D, self.params.k, self.params.dt, 
        self.params.NReal, self.PSDs.shape[0],suffix)
        
        trajPath = os.path.join(folder, trajFilename)
                    
        with open(trajPath, 'w' , newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(self.trajectories)
            
        print("Last trajectories successfully saved to file.")
        
        
        
    def read_PSDs_from_file(self,NIter,prefix,folder='',suffix='', readV0Arr=False,readTrajArr=False):
        '''Read PSD and v0Arr data from file and save data
        to instance'''
        
        import csv
        import os
        
        super().read_PSDs_from_file(NIter,prefix,folder,suffix,readV0Arr)
                        
        if readTrajArr:
            filename = "%s_D%f_k%i_dt%.3f_NReal%i_NIter%i_trajArr%s.csv" %(
                prefix, self.params.D, self.params.k, self.params.dt, 
                self.params.NReal, self.PSDs.shape[0],suffix)
            path = os.path.join(folder, filename)
        
            with open(path, newline='') as csvfile:
                reader = csv.reader(csvfile)
                self.trajectories = np.array([row for row in reader], dtype=float)
        
            print("Most recent trajectories successfully loaded from file.")
        
        
        
        
        

