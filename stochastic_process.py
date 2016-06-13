# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 16:02:13 2015

@author: Daniel Rose

General definition of a stochastic process as python object that can be inherited from.
"""

import numba as nb
import numpy as np
import os


@nb.jit
def PSD_to_frequencies(k, dt, PSD, crop=None):
    '''Function that simulates a frequency series from a given power spectrum
    Input:
    * k   : number of points in time space
    * dt  : time step
    * PSD : power spectrum
    * crop: (optional) crop frequency to shorten calculation time
    Output:
    * fsim: simulated frequency series (as numpy array)'''  

    T = k*dt
    
    if crop: flen = int(crop*k*dt) # length of cropped array
    else: flen = int(k/2+1) # length of frequency array


    fsim = np.empty(flen, dtype=complex) #allocate memory for array
    
    n = np.random.normal(0,1)    # Gaussian random number, mu=0, sigma=1
    fsim[0] = np.sqrt(PSD[0]*T)*n # first (zero-frequency) value must be real    
       
    for i in range(1,flen):
        n = 1/np.sqrt(2) * (np.random.normal(0,1) + 1j*np.random.normal(0,1))
        fsim[i] = np.sqrt(PSD[i]*T)*n
        # could be improved by using numpy cast, after generating all random 
        # numbers in one call
    return fsim

class InitializationError(Exception):
    pass


###############################################################################
######### Definition of main class for stochastic processes ###################
###############################################################################


class StochasticProcess(object):
    '''Class that defines a general stochastic process
        
    Input: 
    
    * signal_PSD    : (optional) power spectrum to be used as signal input
    * params        : dictionary containing parameters
    * process       : an equation defining the core of any stochastic process
    
    todo: 
    * 'update' method, that update parameters and reinitializes process?
    '''
    
    required_keys = ['k', 'dt', 'v0']
    
    def __init__(self, process, signal_PSD=[None], **params):
        '''Method that initializes the process. Parameters in *params* 
        are mapped to attributes of the process.
        '''
        
        if any((key not in params for key in self.required_keys)):
            raise KeyError(
            'Could not initialize object. At least one parameter is missing. \
The following parameters are required: %s' %str(self.required_keys)
            )
        
        self.params = Parameters(**params)     
        
        # choose process
        self.process = process
        # check for input signal
        if np.any(signal_PSD):
            self.use_as_signal_PSD(signal_PSD)
            self.get_signal = self.get_signal_from_PSD
        # use Gaussian white noise if no signal given
        else: 
            self.get_signal = self.get_signal_gaussian_white
    
    def use_as_signal_PSD(self, PSD, normalize=False, suppress_f0=False):
        '''Copy input PSD to self.signal_PSD
        Input:
        * PSD        : power spectrum that is to be used as input
        * normalize  : whether or not the input PSD should normalized
        * suppress_f0: whether or not the 0-f-bin of the PSD is to be suppressed'''
        PSD_copy = np.array(PSD) # ensure that PSD is copied, not harming the original

        if suppress_f0:
            PSD_copy[0] = 0
        if normalize:
            self.signal_PSD = self.get_normalized_PSD(PSD_copy)
        else: 
            self.signal_PSD = PSD_copy
            
    
    def get_signal_gaussian_white(self):
        '''Function that computes an array of Gaussian random numbers.'''
        signal = np.random.normal(0,1,self.params.k)/np.sqrt(self.params.dt)
        return signal
        
    def get_signal_from_PSD(self):
        '''Compute stochastic signal from given PSD
        
        Input: 
        * PSD_norm  : normalized power spectrum to draw signal from        
        Output:
        * signal    : signal to be used as input in stochastic process'''
        
        #check if signal exists        
        try:
            fsim = PSD_to_frequencies(self.params.k, self.params.dt, self.signal_PSD)
        
            signal = self.get_irFFT(fsim, self.params.k)
        
            return signal 
        except AttributeError:
            print("Please define 'signal_PSD' before trying to compute a signal.")
        
    def get_trajectory(self):
        '''Collect a trajectory from process, wrapping all constants 
        and skipping the first few values in order to be independent of the
        initial condition. (Currently no values are skipped.) 

        Input:
        * Signal    : externally defined stochastic signal
        
        Output:
        * traj      : trajectory as numpy array
        
        '''
        
        signal = self.get_signal()
        traj = self.process(self.params.k, self.params.dt, self.params.v0, self.params.D, signal)
        
        return traj
        
    def get_trajectory_from_v0(self, v0):
        '''Collect a trajectory from process, wrapping all constants 
        and skipping the first few values in order to be independent of the
        initial condition. (Currently no values are skipped.) 
        Here, a specific initial value (v0) must be given.

        Input:
        * Signal    : externally defined stochastic signal
        * v0        : trajectory-specific initial condition. 
        
        Output:
        * traj      : trajectory as numpy array
        
        '''
        
        signal = self.get_signal()
        traj = self.process(self.params.k, self.params.dt, v0, self.params.D, signal)
        
        return traj

 
    def get_times(self):
        '''Compute time axis.'''
        dt = self.params.dt
        k = self.params.k
        T = k*dt
        t = np.linspace(0,T-dt,k)
        return t

    def get_frequencies(self):
        '''Produce a frequency array fitting to the 
        Fourier transform definition.'''
        
        freq = np.fft.rfftfreq(self.params.k, self.params.dt)  
        return freq

    
    def get_rFFT(self, traj, **kwargs):
        '''Calculate the FFT of one or multiple real-valued trajectories
        
        Input:
        * traj  : trajectory to be Fourier transformed
        
        Output:
        * ft    : array containing frequency'''
        
        ft = np.fft.rfft(traj, self.params.k, **kwargs)*self.params.dt #normalize with dt
        
        return ft
        
    def get_irFFT(self, ft, k, **kwargs):
        '''Calculate trajectories as the inverse FFT of one or multiple hermitian 
        frequency series '''
        
        traj = np.fft.irfft(ft/self.params.dt, k, **kwargs) #remove normalization
        return traj
        
    def get_df(self):
        '''Compute frequency step'''
        return 1/(self.params.k*self.params.dt)
        
    def get_PSD(self, NReal, **kwargs):
        '''Compute a power spectrum from multple trajectories
        
        Input:
        * NReal     : number of realizations to compute
        
        Output:
        * PSD   : power spectrum
        ''' 
        np.random.seed()
        PSD = np.zeros(int(self.params.k/2+1))       
        for i in range(NReal):
            traj = self.get_trajectory()
            ft = self.get_rFFT(traj)
            PSD += abs(ft)**2
        
        PSD /= NReal*self.params.k*self.params.dt
        
        return PSD
    
        
    
    def get_normalized_PSD(self, PSD):
        '''Compute normalized PSD.
        Normalization: Integral_-inf^inf PSD*df = 1
        '''
        df = self.get_df()
        
        # correction for zero-frequency bin
        # PSDnorm[0] = PSDnorm[1]
        
        norm = df*PSD[0] + 2*df*PSD[1:].sum()
        # more accurate: trapezoidal rule?
        # norm = 2*np.trapz(PSDnorm, dx=df) 
        #2 since I am only using half of the spectrum
        
        PSDnorm = np.array(PSD)/norm # ensure copy of original
    
        return PSDnorm
        
    def get_switching_rate(self, traj, lower=-0.5, upper=0.5):
        '''Calculate switching rate 
        Input:
        * index     : index of the iteration, the input PSD is taken from
        Output:
        * r         : average switching rate of the trajectory
        * stdev     : standard deviation of the average r
        '''
        
        switching_points = self.get_switching_points(traj, lower, upper)
        
        if np.any(switching_points):
            T = self.params.k*self.params.dt
            rate = len(switching_points)/T
            # might as well subtract the initial setup time, but well...
            return rate
        else: 
            #print("No jumps, no jump rate")
            return False
    
            
    def get_switching_points(self, traj, lower=-0.5, upper=0.5):
        '''Obtain switching points between two levels of a given trajectory.
        Input: 
        * traj              : trajectory of points
        * lower             : lower threshold
        * upper             : upper threshold
        Output: 
        * switchingPoints   : array containing the indices where the trajectory
                              switched between the two levels
        '''
        
        # find starting level
        
        start = -1        
        for i, value in enumerate(traj):
            if value >= upper:
                pos = 1
                start = i
                break
            elif value <= lower:
                pos = -1
                start = i
                break
            else:
                pos = False # could equivalently be 0
                pass

        if not pos:
            print("None of the thresholds were reached.")
            crossings = False
            
        else: 
            # and now find all switches
            #level = [False,1,-1]
            threshold = [0,lower,upper]
            crossings = []
            for i, value in enumerate(traj[start:]):
                if (value*pos <= threshold[pos]*pos):
                    crossings.append(i)
                    pos = (-1)*pos
        
        return crossings 
        
###############################################################################
######################## PSD iteration Class ##################################
###############################################################################

class StochasticProcess_with_PSD_Iteration(StochasticProcess):
    '''Base class for a stochastic process, which uses its power spectrum
    again as input in an iterative procedure. Data can be loaded from file, 
    saved to file and completed, if not yet complete.'''
    
    required_keys = StochasticProcess.required_keys + ["NReal"]
    
    def __init__(self, process, PSDs=[None], normalize=False,
                 skip = 0, signalInit=True, **params):
        '''Initializing the general stochastic process plus some extras
        Input:
        * process   : a specific process class (assumed to be a subclass to the superclass
                        with a process function defined)
        * normalize : whether or not the output spectra of each generation should 
                      be renormalized for use in the next generation.
        * skip      : number of time points that should be skipped in the trajectories
                      in order to cut out transients in the beginning.
        '''        
        
        
        # update required parameters
        #for key in SpecProcess.required_keys:
        #    if key not in self.required_keys:
        #        self.required_keys.append(key)
        
        
        # initialize superclass
        #process = SpecProcess.process
        super().__init__(process, **params)


        self.update_v0Arr()
            
        self.update_PSDs(PSDs)
        
        self.params.update(normalize = normalize, skip = skip)
        
        # define signal source
        self.get_signal = self.get_signal_from_PSD
        if signalInit:
            self.use_as_signal_PSD(self.PSDs[-1,:],self.params.normalize)
    
    
    def update_v0Arr(self):
        '''Check integrity of data for initial conditions and transfer them to 
        'v0Arr' array of initial conditions. In this case for a parallel simulation
        scheme.'''
        
        # initialize v0Arr
        self.v0Arr = np.empty((1,self.params.NReal))
        
        if np.shape(self.params.v0) != (): # not scalar
            if np.shape(self.params.v0)[-1] != self.v0Arr.shape[-1]:
                raise InitializationError("Number of initial conditions (v0) and \
number of realizations (NReal) do not coincide. Initialization failed.")
            elif np.shape(np.shape(self.params.v0)) == (2,): # 2D array
                self.v0Arr = np.array(self.params.v0)
            elif np.shape(np.shape(self.params.v0)) == (1,): # 1D array
                self.v0Arr[0,:] = self.params.v0[:]
            else:
                raise InitializationError("Dimension of initial conditions (v0) is inconsistent. \
Initialization failed.")

        else: # scalar IC
            # pass single initial condition to array of initial conditions
            self.v0Arr.fill(self.params.v0)

           
           
    def update_PSDs(self, PSDs):
        '''Check integrity of data for previously calculated PSDs and transfer 
        them to 'PSDs': array of power spectra, one for each generation'''
        
        # pass previously calculated PSDs to internal variable
        # initialize internal array
        self.PSDs = np.empty((1,int(self.params.k/2+1)))
        
        if np.any(PSDs): # anything given?
            if PSDs.shape[-1] != self.PSDs.shape[-1]:
                raise InitializationError("Last dimension of PSDs (= %i) and \
number of samples (k/2+1 = %i) do not coincide. Initialization failed." 
                                          %(np.shape(PSDs)[-1], self.PSDs.shape[-1]))
            else:
                if np.shape(PSDs.shape) == (2,): # more than one generation given
                    self.PSDs = np.array(PSDs) # copy  
                elif np.shape(PSDs.shape) == (1,): # only input given
                    self.PSDs[0] = np.array(PSDs) # copy
                else: 
                    raise InitializationError("Dimension of 'PSDs' is inconsistent. \
Initialization failed.")
                
                # normalize input PSDs if required
                #if normalize:
                #    for i in range(PSDs.shape[0]):
        else: # nothing given
            #self.PSDs.fill(self.params.dt)
            self.PSDs.fill(1)
            # Gaussian white noise as input for first generation
        
        
    def get_trajectory(self, v0):
        '''Collect a trajectory from process, wrapping all constants 
        and skipping the first few values in order to be independent of the
        initial condition. (Currently no values are skipped.) 
        Here, a specific initial value (v0) must be given.

        Input:
        * Signal    : externally defined stochastic signal
        * v0        : trajectory-specific initial condition. 
        
        Output:
        * traj      : trajectory as numpy array
        '''
        
        signal = self.get_signal()
        #signal -= signal.mean() # artificially set mean of signal to zero, 
                                 # would need to be renormalized
        traj = self.process(self.params.k, self.params.dt, v0, self.params.D, signal)
        
        return traj
    
    
    
    def get_trajectory_periodic(self, v0):
        '''Collect a trajectory from process, wrapping all constants 
        and skipping the first few values in order to be independent of the
        initial condition. (Currently no values are skipped.) 
        Here, a specific initial value (v0) must be given.

        Input:
        * Signal    : externally defined stochastic signal
        * v0        : trajectory-specific initial condition. 
        
        Output:
        * traj      : trajectory as numpy array
        '''
        
        signal = self.get_signal()

        newV0 = self.process(self.params.k, self.params.dt, v0, self.params.D, signal)[-1]
        traj = self.process(self.params.k, self.params.dt, newV0, self.params.D, signal)
        
        return traj
    
    
    def get_trajectory_with_index(self, v0, index):
        '''Collect a trajectory of a specified iteration (index)'''
        
        self.use_as_signal_PSD(self.get_PSD(index),self.params.normalize)
        return self.get_trajectory(v0)
        
    
    def get_PSD(self, index, getAll=False, normed=False, repeat=False, 
                suppress_f0=False, **kwargs):
        '''Collect the power spectrum of a specified iteration (index).
        If it exists, it is returned, otherwise it is calculated.
        Input:
        * index     : iteration index to pull PSD from
        Output:
        * PSD       : power spectrum of specified iteration'''
        
        if index < self.PSDs.shape[0]:
            # just pull it
            pass
        else:
            # check initial conditions
            if self.PSDs.shape[0] > self.v0Arr.shape[0]:
                print("More PSDs are given than initial conditions (v0Arr). Filling Initial conditions up with latest values found.")
                foo = self.v0Arr.shape[0] # save number of existent generations      
                self.v0Arr.resize((self.PSDs.shape[0],self.v0Arr.shape[-1]))
                self.v0Arr[foo:] = self.v0Arr[foo-1]
            # calculate it
            print('Starting computations...')
            for i in range(self.PSDs.shape[0]-1, index):
                self.compute_next_PSD_iteration(self.v0Arr[i,:],repeat,suppress_f0,**kwargs)

                print('Finished iteration %i of %i' %(i+1, index))
                
            # return last computed PSD
        
        if normed: # might not be save to use
            PSDsNormed = np.empty_like(self.PSDs)
            for i in range(self.PSDs.shape[0]):
                PSDsNormed[i,:] = self.get_normalized_PSD(self.PSDs[i,:])
            return PSDsNormed
        else:
            if getAll: return np.array(self.PSDs) #ensure copy of data
            else: return np.array(self.PSDs[index,:])#ensure copy of data
            
            
    def compute_next_PSD_iteration(self, v0Arr,repeat,suppress_f0,**kwargs):
        '''Compute a new iteration for the power spectrum.
        Remark:
        * specify kwarg 'suppress_f0' as True, if 0-f-bin of PSD should be 
        suppressed (set to 0)
        ''' 
        
        self.use_as_signal_PSD(self.PSDs[-1,:],self.params.normalize, suppress_f0, **kwargs)
        PSD, lastValArr = self.compute_PSD(v0Arr,repeat)
        # lastValArr is supposed to be an array containing the last values
        # of all trajectories of the current generation.
        # in case of series simulation instead of parallel, 
        # the initial conditions are just returned, unless otherwise specified
        
        # save results to internal arrays
        self.v0Arr = np.concatenate((self.v0Arr,lastValArr), axis=0)
        self.PSDs = np.concatenate((self.PSDs,PSD), axis=0)

    
    def compute_PSD(self, v0Arr,repeat):
        '''Compute a power spectrum from multple trajectories.
        Here, the initial condition 'v0' is assumed to be an array.
        Implementation for a parallel simulation scheme (vector operations)
        
        Input:
        * v0Arr     : array containing initial conditions for each trajectory
                        now assumed to be internal variable
        
        Output:
        * PSD   : power spectrum
        * lastValArr : array containing last value of each trajectory (for IC)
        ''' 
        np.random.seed()

        # initialize new PSD and lastValArr
        PSD = np.zeros((1,int(self.params.k/2 + 1)))
        lastValArr = np.empty((1,self.params.NReal))
        
        # periodic trajectories:
        if repeat:
            get_trajectory = self.get_trajectory_periodic
        else:
            get_trajectory = self.get_trajectory
                
        # set up ensemble mean of trajectories
        ens_mean_f0 = 0
        # sum up all FTs
        for i in range(self.params.NReal):
            traj = get_trajectory(v0Arr[i])
            lastValArr[0,i] = traj[-1]
            ft = self.get_rFFT(traj[self.params.skip:]) #skip first few points, if desired (might not be stable, not tested, not used)
            ens_mean_f0 += ft[0].real # has zero imaginary part by construction
            PSD += abs(ft)**2
            
        # normalize
        ens_mean_f0 /= self.params.NReal # div by ensemble size
        T = self.params.k*self.params.dt
        PSD /= self.params.NReal*T
        # subtract ensemble mean
        PSD[0,0] -=  ens_mean_f0*ens_mean_f0/T # 1/T * <x>^2, should be positive
        # failsafe method, (very small numbers tend to become negative... in a way)
        if PSD[0,0] < 0.:
            PSD[0,0] = 0.
        return PSD, lastValArr

    
    def get_variance(self, index, *args, **kwargs):
        '''Compute variance of a particular PSD by integrating it'''
        PSD = self.get_PSD(index, *args, **kwargs)
        df = self.get_df()
        #variance = df*(PSD[0]+ 2*PSD[1:].sum()) # naive integral
        variance = 2*np.trapz(PSD, dx=df) # trapezoidal integral
        return variance
    
    def get_variances(self, nmax, *args, **kwargs):#, integType='trapz'):
        '''Evaluate the variance of all PSDs that have been saved'''
        
        
        
        varArr = np.empty(nmax)
        nArr = np.arange(1,nmax+1)
        
        for i in range(nmax):
            varArr[i] = self.get_variance(i+1,*args,**kwargs)
            
        return nArr, varArr
        
        
    def get_correlation_function(self,index,normalize=False,zeropadding=False,**kwargs):
        '''Compute autocorrelation function of specific PSD defined by 'index'
        '''
        
        PSD = self.get_PSD(index, **kwargs)
        CF = self.get_irFFT(PSD, self.params.k)
        CF = CF[:self.params.k]
        t = self.get_times()[:self.params.k]
        
        if normalize:
            CF = CF/CF[0]
        
        if zeropadding:
            for i in range(len(CF)):
                if CF[i] < 0:
                    CF[i:] = 0
                    break
                else: pass
                    
        return t, CF
        
    def get_correlation_time(self,index,*args,**kwargs):
        '''Compute the correlation time of specific generation'''
        
        PSD = self.get_PSD(index,*args,**kwargs)        
        df = self.get_df()

        PSD_sq_int = 2*np.trapz(PSD*PSD, dx=df)
        PSD_int = 2*np.trapz(PSD, dx=df)
        tau = 2*np.pi * PSD_sq_int/(PSD_int*PSD_int)     

        return tau
        
    def get_correlation_times(self,nmax,*args,**kwargs):
        '''Compute an array of correlation times up to given maximum
        generation index 'nmax'''
        
        tauArr = np.empty(nmax)
        nArr = np.arange(1,nmax+1)
        
        for i in range(nmax):
            tauArr[i] = self.get_correlation_time(i+1,*args,**kwargs)
        
        return nArr, tauArr
        
    def get_switching_rates(self, v0=None, nReal=10, lower=-0.5, upper=0.5):
        '''Calculate jump rates for all stored PSDs
        Input:
        * v0        : Initicial condition to start trajectories from
        * NReal     : (default = 10) number of realizations to average over
        * lower     : lower threshold to register switching (should be <0)
                      (default = -0.5)
        * upper     : upper threshold to register switching (should be >0)
                      (default = 0.5)
        Output:
        * nArr          : array containing the generation indeces
        * rateMeans  : means of jump rates
        * rateStds   : standard deviations of jump rates
        '''

        nGen = self.PSDs.shape[0]  # no. of generations + input PSD (NIter + 1)      
        
        
        if v0: # uniform IC for all trajectories
            v0Arr = np.ones((nGen, nReal))*v0
        else: # random sample of IC from internal v0Arr
            v0Arr = np.empty((nGen, nReal))
            for genIndex in range(nGen):
                v0Arr[genIndex,:] = np.random.choice(self.v0Arr[genIndex,:],nReal)
                
                
        rateArr = np.zeros((nGen,nReal))
        rateMeans = np.zeros(nGen)
        rateStds = np.zeros(nGen)
        # main loop
        for genIndex in range(nGen): # generations
            for realIndex in range(nReal): # realizations
                traj = self.get_trajectory_with_index(v0Arr[genIndex,realIndex],
                                                      genIndex)
                rateArr[genIndex,realIndex] = super().get_switching_rate(traj)

            print("Finished generation %i" %genIndex)
            rateMeans[genIndex] = rateArr[genIndex,:].mean()
            rateStds[genIndex] = rateArr[genIndex,:].std()
            
        nArr = np.arange(nGen)
        
        return nArr, rateMeans, rateStds
        
    def get_switching_rate(self,index, v0=None, nReal=10, lower=-0.5, upper=0.5):
        '''Calculate jump rate for a single stored PSD
        Input:
        * v0        : Initicial condition to start trajectories from
        * NReal     : (default = 10) number of realizations to average over
        * lower     : lower threshold to register switching (should be <0)
                      (default = -0.5)
        * upper     : upper threshold to register switching (should be >0)
                      (default = 0.5)
        Output:
        * nArr          : array containing the generation indeces
        * rateMeans  : means of jump rates
        * rateStds   : standard deviations of jump rates
        '''

        
        
        if v0: # uniform IC for all trajectories
            v0Arr = np.ones(nReal)*v0
        else: # random sample of IC from internal v0Arr
            v0Arr = np.random.choice(self.v0Arr[index,:],nReal)
                
                
        rateArr = np.zeros(nReal)
        # main loop
        for realIndex in range(nReal): # realizations
            traj = self.get_trajectory_with_index(v0Arr[realIndex], index)
            rateArr[realIndex] = super().get_switching_rate(traj)

        rateMean = rateArr.mean()
        rateStd = rateArr.std()
                    
        return rateMean, rateStd
                      
            

    def write_PSDs_to_file(self, prefix, folder='', suffix=''):
        '''Write computed PSDs to file as well as the v0Arr'''
        
        import csv
        
        filename = "%s_D%f_k%i_dt%.3f_NReal%i_NIter%i_PSDs%s.csv" %(
        prefix, self.params.D, self.params.k, self.params.dt, 
        self.params.NReal, self.PSDs.shape[0],suffix)
        
        v0ArrFilename = "%s_D%f_k%i_dt%.3f_NReal%i_NIter%i_v0Arr%s.csv" %(
        prefix, self.params.D, self.params.k, self.params.dt, 
        self.params.NReal, self.PSDs.shape[0],suffix)
        
        path = os.path.join(folder, filename)
        v0ArrPath = os.path.join(folder, v0ArrFilename)
        
        with open(path, 'w' , newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(self.PSDs)
            
        with open(v0ArrPath, 'w' , newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(self.v0Arr)
            
        print("PSDs and v0Arr have successfully been saved to file.")
    
    
    def read_PSDs_from_file(self,NIter,prefix,folder='',suffix='', readV0Arr=False):
        '''Read PSD and v0Arr data from file and save data
        to instance'''
        
        import csv
        
        filename = "%s_D%f_k%i_dt%.3f_NReal%i_NIter%i_PSDs%s.csv" %(
            prefix, self.params.D, self.params.k, self.params.dt, 
            self.params.NReal, NIter,suffix)
        path = os.path.join(folder, filename)
        
        with open(path, newline='') as csvfile:
            reader = csv.reader(csvfile)
            self.PSDs = np.array([row for row in reader], dtype=float)
            
        print("PSDs successfully loaded from file.")
            
        if readV0Arr:
            filename = "%s_D%f_k%i_dt%.3f_NReal%i_NIter%i_v0Arr%s.csv" %(
                prefix, self.params.D, self.params.k, self.params.dt, 
                self.params.NReal, self.PSDs.shape[0],suffix)
            path = os.path.join(folder, filename)
        
            with open(path, newline='') as csvfile:
                reader = csv.reader(csvfile)
                self.v0Arr = np.array([row for row in reader], dtype=float)
        
            print("Initial conditions (v0Arr) successfully loaded from file.")
            
            
    def write_data_to_file(self, data, dataName, prefix, folder='', suffix=''):
        '''A generalized save function that save some arbitrary data under a 
        given name'''
        
        import csv
        
        filename = "%s_D%f_k%i_dt%.3f_NReal%i_NIter%i_%s%s.csv" %(
        prefix, self.params.D, self.params.k, self.params.dt, 
        self.params.NReal, self.PSDs.shape[0],dataName,suffix)
        
        path = os.path.join(folder, filename)
        
        if len(np.shape(data)) == 1:
            
            with open(path, 'w' , newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(data)
        else:
            with open(path, 'w' , newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(data)
        
            
        print("Successfully saved %s to file." %dataName)
        
    def read_data_from_file(self,dataName,NIter,prefix,folder='',suffix=''):
        '''Generalized read function that reads arbitrary data and returns it.
        '''
        
        import csv
        
        filename = "%s_D%f_k%i_dt%.3f_NReal%i_NIter%i_%s%s.csv" %(
        prefix, self.params.D, self.params.k, self.params.dt, 
        self.params.NReal,NIter,dataName,suffix)
        
        path = os.path.join(folder, filename)
        
        with open(path, newline='') as csvfile:
            reader = csv.reader(csvfile)
            data = np.array([row for row in reader], dtype=float)
            
            
        print("Successfully loaded %s from file." %dataName)
        return data
        
###############################################################################
################## Parameter Class ############################################
        
class Parameters(object):
    '''Class that defines a collection of parameters, 
    returns prints keys and parameters via __str__ and 
    returns parameter dictionary via __call__'''
    
    def __init__(self, **params):
        '''Collect all given parameters as attributes and remember keys.'''
        
        self.keys = []
        self.update(**params)
    
    def update(self, **params):
        '''Method that saves parameters as attributes and remembers keys, 
        tries to check consistency.'''
        
        for key, item in params.items():
            setattr(self, key, item)
            
            if key not in self.keys:
                self.keys.append(key)
                
    def __call__(self):
        '''Return dictionary of parameters'''
        
        myparams = {}
        for key in self.keys:
            myparams[key] = getattr(self, key)
            
        return myparams
    
    def __str__(self):
        
        itemlist = ("%s = %s" %(key, item) for key, item in self().items())
        itemlist = '\n'.join(itemlist)
        
        output = "The following parameters are stored: \n%s" %itemlist 
        return output
        
    
