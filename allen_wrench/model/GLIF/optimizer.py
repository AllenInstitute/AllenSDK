import logging

from scipy import *
import numpy as np
import copy
import matplotlib.pyplot as plt
import time
from scipy.optimize import fminbound, fmin
from scipy.optimize import minimize
import json
from numpy import ndarray, array
import error_functions
import os.path
import warnings

from uuid import uuid4
import random


class GLIFOptimizer(object):
    def __init__(self, experiment, dt, 
                 outer_iterations, inner_iterations, 
                 sigma_outer, sigma_inner,
                 param_fit_names, stim, 
                 error_function_name, neuron_num, 
                 xtol, ftol, 
                 internal_iterations, init_params=None):

        self.experiment = experiment
        self.dt = dt
        self.outer_iterations = outer_iterations
        self.inner_iterations = inner_iterations
        self.start_time = None
        self.init_params = init_params
        self.sigma_outer = sigma_outer
        self.sigma_inner = sigma_inner
        self.param_fit_names = param_fit_names
        self.stim = stim
        self.error_function = error_functions.get_error_function_by_name(error_function_name)
        self.neuron_num = neuron_num
        self.xtol = xtol
        self.ftol = ftol
        self.internal_iterations = internal_iterations

        self.iteration_info = [];

        if self.init_params is None:
            self.init_params = np.ones(experiment.neuron_parameter_count())

    def to_dict(self):
        return {
            'dt': self.dt,
            'outer_iterations': self.outer_iterations,
            'inner_iterations': self.inner_iterations,
            'init_params': self.init_params,
            'sigma_outer': self.sigma_outer,
            'sigma_inner': self.sigma_inner,
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
            
    def randomize_parameter_values(self, values, sigma):
        values = np.array(np.random.normal(values, sigma))

        # values might not have a shape if it's a single element long, depending on your numpy version
        if not values.shape:
            values = np.array([values])
        values[values<0] = 0
        return values
        
    def run_many(self, iteration_finished_callback=None):
        params = self.init_params
        self.start_time = time.time()

        for outer in range(0, self.outer_iterations):  #outerloop 
            for inner in range(0, self.inner_iterations):  #innerloop
                iteration_start_time = time.time()

                # run the optimizer once.  first time is always the passed initial conditions.
                opt = self.run_once(params)
                xopt, fopt = opt[0], opt[1]

                logging.info('fmin took %f secs, %f mins, %f hours' %  (time.time() - iteration_start_time, (time.time() - iteration_start_time)/60, (time.time() - iteration_start_time)/60/60))

                self.iteration_info.append({
                    'in_params': params.tolist(),
                    'out_params': xopt.tolist(),
                    'error': float(fopt)
                })

                if iteration_finished_callback is not None:
                    iteration_finished_callback(self, outer, inner)

                # randomize the best fit parameters
                params = self.randomize_parameter_values(xopt, self.sigma_inner)
                #params = xtol*(1-self.eps/2+self.eps*np.random.random(len(params),))

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
                #tempTimeIterStart=time.time()
                #(voltage_list, threshold_list, AScurrentMatrix_list, gridSpikeTime_list, interpolatedSpikeTime_list, \
                #    gridSpikeIndex_list, interpolatedSpikeVoltage_list, interpolatedSpikeThreshold_list) = \
                #    self.experiment.run_base_model(xtol)

                #timeFor1Iter=time.time()-tempTimeIterStart  

            # outer loop uses the outer standard deviation to randomize the initial values
            params = self.randomize_parameter_values(self.init_params, self.sigma_outer)


        # get the best one!
        min_error = float("inf")
        min_i = -1
        for i, info in enumerate(self.iteration_info):
            if info['error'] < min_error:
                min_error = info['error']
                min_i = i

        best_params = self.iteration_info[min_i]['out_params']

        self.experiment.set_neuron_parameters(best_params)

        logging.info('done optimizing')
        return best_params, self.init_params

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
#        fmin(func, x0, args=(), xtol=1e-4, ftol=1e-4, maxiter=None, maxfun=None, full_output=0, disp=1, retall=0, callback=None):

        return fmin(self.error_function, param0, args=(self.experiment,),xtol=self.xtol, ftol=self.ftol,  maxiter=self.internal_iterations, maxfun=self.internal_iterations, retall=1,full_output=1, disp=1)

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
#        iteration_start_time = time.time()
#        xopt = minimize(self.error_function, param0, args=(self.experiment,), method='Newton-CG', jac=f_prime_constructor(self.error_function), callback=mycallback_ncg, options=options, tol=eps)
#        print 'Newton-CG method took', (time.time()-iteration_start_time)/60., 'seconds'
#
##        print 'Using Nelder-Mead method'
##        iteration_start_time = time.time()
##        xopt = minimize(self.error_function, param0, args=(self.experiment,), method='Nelder-Mead', callback=mycallback_nm, options=options, tol=eps)
##        print 'Nelder-Mead method took', (time.time()-iteration_start_time)/60., 'seconds'
#
#        print xopt
        return xopt, fopt

