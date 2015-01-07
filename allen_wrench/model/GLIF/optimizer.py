from scipy import *
import numpy as np
import copy
import matplotlib.pyplot as plt
from time import time
from scipy.optimize import fminbound, fmin
from scipy.optimize import minimize
import json
from numpy import ndarray, array
import error_functions
import os.path
import warnings

from uuid import uuid4
import random


class GLIFOptimizer(object): #not sure why object is in here
    def __init__(self, experiment=None, dt=None, upper_bounds=None, lower_bounds=None,
                 inner_loop=None, start_time=None, eps=None, param_fit_names=None, stim=None, 
                 error_function_name=None, neuron_num=None, xtol=None, ftol=None, save_file_name=None,
                 internal_iterations=None, internal_func=None):

        self.experiment = experiment
        self.dt = dt
        self.upper_bounds = upper_bounds
        self.lower_bounds = lower_bounds
        self.inner_loop = inner_loop
        self.start_time = start_time
        self.save_file_name = save_file_name
        self.eps = eps
        self.param_fit_names = param_fit_names
        self.stim = stim
        self.error_function = error_functions.get_error_function_by_name(error_function_name)
        self.neuron_num = neuron_num
        self.xtol = xtol
        self.ftol = ftol
        self.internal_iterations = internal_iterations
        self.internal_func = internal_func

        self.iteration_info = [];

    def to_dict(self):
        return {
            'dt': self.dt,
            'upper_bounds': self.upper_bounds,
            'lower_bounds': self.lower_bounds,
            'inner_loop': self.inner_loop,
            'start_time': self.start_time,
            'eps': self.eps,
            'param_fit_names': self.param_fit_names,
            'neuron_num': self.neuron_num,
            'xtol': self.xtol,
            'ftol': self.ftol,
            'internal_iterations': self.internal_iterations,
            'iteration_info': self.iteration_info,
            'init_voltage': self.experiment.init_voltage,
            'init_threshold': self.experiment.init_threshold,
            'init_AScurrents': self.experiment.init_AScurrents
        }
            
    def generateInitParamGuess(self):
        return self.lower_bounds+(self.upper_bounds-self.lower_bounds)*np.random.random(len(self.upper_bounds)) #specify random initial condition  #THIS WILL HAVE TO BE ADAPTED FOR VARIOUS PARAMETERS!!!!!!!!!!!!!!!!!!!!!!!!!!!

    def run_many(self, outloop, iteration_finished_callback=None):
        initial_params = None

        for o in range(0,outloop):  #outerloop 
            params = self.generateInitParamGuess()

            if initial_params is None:
                initial_params = params

            for s in range(0,self.inner_loop):  #innerloop
                #I think this is the bug is here this should be indicies not times and this should be the target not ref ISI
                # bestFitParam.append(fmin(TRDMinRoutine, param0[s], args=(var_in, peakInd, ISI, stim), xtol=.0001, ftol=.0001, maxiter=2000, full_output=1 ))  #THIS IS THE OLD WAY
                # bestFitParam.append(fmin(self.optimizer.thresh_routine, param0[s], args=(self.fit_names, self.var_in, self.peakInd, self.ISI, self.stim), xtol=.0001, ftol=.0001, maxiter=2, full_output=1 ))

                start_time = time()
                opt = self.run_once(params)
                xtol, ftol = opt[0], opt[1]
                print 'fmin took', time()-self.start_time, "seconds",  (time()- self.start_time)/60, 'mins', (time()-self.start_time)/60/60, 'hours'

                self.iteration_info.append({
                    'in_params': params.tolist(),
                    'out_params': xtol.tolist(),
                    'error': float(ftol)
                })

                # randomize the best fit parameters
                params = xtol*(1-self.eps/2+self.eps*np.random.random(len(params),))

                if iteration_finished_callback is not None:
                    iteration_finished_callback(self, o, s)

                #cant figure out how to run the model so I might have to just save the results and then run them else where
                #(voltage, gridtime)=self.model.neuron.runModel(self, self.neuron.init_voltage, self.neuron.init_threshold, self.neuron.init_AScurrents, self.stim)
        #        if (plot==1):
        #            print 'in plotting routine'
        #            fitvoltage=[]        
        #            #----Here do I need to set the variables in the initialization (p.coeffdef=1) or should they automatically be the correct ones????
        #            fitvoltage=self.p.runModel(stim, voltage)
        #            #add a bit of randomness to the best fits and make sure that it once again converges their
        #            plt.figure(o+plotNumStart)
        #            if s==0:
        #                plt.subplot(numFminIter+1,1,1)
        #                plt.plot(stim) #plot stimulus 4
        #                plt.subplots_adjust(wspace=.8, hspace=.8)                
        #            plt.subplot(numFminIter+1,1,s+2)
        #            plt.subplots_adjust(wspace=.8, hspace=.8)
        #            plt.plot(voltage, 'b')
        #            plt.plot(fitvoltage*1000, 'r')
        #            plt.title(['Param', bestFitParam[s][0], 'TRD', bestFitParam[s][1]], fontsize=12)
        #            plt.show(block=False)
                
                #---Calculate the other fitness functions
                
                '''Add other fitness functions here
                first gotta calculate the spike trains again
                also look and see what the difference between
                the size of a pickle file and a text file is to 
                decide how to save stimulus and traces'''
                
                #Take the current best values and run the program again.
                #tempTimeIterStart=time()
                #(voltage_list, threshold_list, AScurrentMatrix_list, gridSpikeTime_list, interpolatedSpikeTime_list, \
                #    gridSpikeIndex_list, interpolatedSpikeVoltage_list, interpolatedSpikeThreshold_list) = \
                #    self.experiment.run_base_model(xtol)

                #timeFor1Iter=time()-tempTimeIterStart  

        # get the best one!
        min_error = float("inf")
        min_i = -1
        for i, info in enumerate(self.iteration_info):
            if info['error'] < min_error:
                min_error = info['error']
                min_i = i

        print 'done optimizing'
        return self.iteration_info[min_i]['out_params'], initial_params

    def run_once_bound(self, low_bound, high_bound):
        '''
        @param low_bound: a scalar initial guess for the optimizer
        @param high_bound: a scalar high bound for the optimizer
        @return: tuple including parameters that optimize function and value - see fmin docs
        '''                
        return fminbound(self.error_function, low_bound, high_bound, args=(self.experiment,), maxfun=200, full_output=True )
        #Note is defined in the top level script
    
  
    def run_once(self, param0):
        '''
        @param param0: a list of the initial guesses for the optimizer
        @return: tuple including parameters that optimize function and value - see fmin docs
        '''       
        print 'why isn''t this thing printing the iteration values!?' 
#        fmin(func, x0, args=(), xtol=1e-4, ftol=1e-4, maxiter=None, maxfun=None, full_output=0, disp=1, retall=0, callback=None):

        return fmin(self.error_function, param0, args=(self.experiment,self.save_file_name),xtol=self.xtol, ftol=self.ftol,  maxiter=self.internal_iterations, maxfun=self.internal_func, retall=1,full_output=1, disp=1)

#        #Note is defined in the top level script
#        def mycallback_ncg(xk):
#            print 'Using Newton-CG method, xk: ', xk
#        def mycallback_nm(xk):
#            print 'Using Nelder-Mead method, xk: ', xk
#        eps=1e-15
#        options={}
#  #      options['avextox']=eps
#        options['maxiter']=500
##        options['full_output']=True
##        options['disp']=True
##        options['retall']=True
#
#        print 'Using Newton-CG method'
#        start_time = time()
#        xopt = minimize(self.error_function, param0, args=(self.experiment,), method='Newton-CG', jac=f_prime_constructor(self.error_function), callback=mycallback_ncg, options=options, tol=eps)
#        print 'Newton-CG method took', (time()-start_time)/60., 'seconds'
#
##        print 'Using Nelder-Mead method'
##        start_time = time()
##        xopt = minimize(self.error_function, param0, args=(self.experiment,), method='Nelder-Mead', callback=mycallback_nm, options=options, tol=eps)
##        print 'Nelder-Mead method took', (time()-start_time)/60., 'seconds'
#
#        print xopt
        return xopt, fopt

