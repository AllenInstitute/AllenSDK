import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse
from scipy.linalg import solve_banded
from scipy.stats import linregress as linregress
import logging

def get_diagonals_from_sparse(mat):
    ''' Returns a dictionary of diagonals keyed by offsets

    Parameters
    ----------
    mat: scipy.sparse matrix

    Returns
    -------
    dictionary: diagonals keyed by offsets
    '''

    mat_dia = mat.todia()  #make sure the matrix is in diagonal format

    offsets = mat_dia.offsets
    diagonals = mat_dia.data

    mat_dict = {}

    for i,o in enumerate(offsets):
        mat_dict[o] = diagonals[i]

    return mat_dict

def ab_from_diagonals(mat_dict):
    ''' Constructs value for scipy.linalg.solve_banded

    Parameters
    ----------
    mat_dict: dictionary of diagonals keyed by offsets

    Returns
    -------
    ab: value for scipy.linalg.solve_banded
    '''
    offsets = mat_dict.keys()
    l = -np.min(offsets)
    u = np.max(offsets)

    T = mat_dict[offsets[0]].shape[0]

    ab = np.zeros([l+u+1,T])

    for o in offsets:
        index = u-o
        ab[index]=mat_dict[o]

    return ab

def error_calc(F_M, F_N, F_C, r):

    er = np.sqrt(np.mean(np.square(F_C-(F_M-r*F_N))))/np.mean(F_M)

    return er

def error_calc_outlier(F_M, F_N, F_C, r):

    std_F_M = np.std(F_M)
    mean_F_M = np.mean(F_M)
    ind_outlier = np.where(F_M > mean_F_M + 2.*std_F_M)

    er = np.sqrt(np.mean(np.square(F_C[ind_outlier]-(F_M[ind_outlier]-r*F_N[ind_outlier]))))/np.mean(F_M[ind_outlier])

    return er

def ab_from_T(T, lam, dt):
    Ls = -sparse.eye(T-1,T,format='csr') + sparse.eye(T-1,T,1,format='csr')  #using csr because multiplication is fast
    Ls /= dt
    Ls2 = Ls.T.dot(Ls)
    
    M = sparse.eye(T) + lam*Ls2
    mat_dict = get_diagonals_from_sparse(M)
    ab = ab_from_diagonals(mat_dict)
    
    return ab

class NeuropilSubtract (object):

    def __init__(self, T, lam=0.1, dt=1.0):
        self.T = T
        self.r = None
        self.ab = ab_from_T(T, lam, dt)
        
    def set_F(self,F_M, F_N):
        ''' Internal initializatin routine
        '''
        self.F_M = F_M
        self.F_N = F_N
        self.F_M_crossval = F_M_crossval
        self.F_N_crossval = F_N_crossval

    
    #def fit_grad_desc_early_stop(self,r_init=0.001,learning_rate=0.1):
    def fit_grad_descent(self, r_init=0.5, learning_rate=100.0, 
                         max_iterations=10000, min_delta_r=0.00001, max_error=0.2):
        ''' Calculate fit using gradient decent, as opposed to iteratively
        computing exact solutions
        '''
        delta_r = 1
        r = r_init

        r_list = []
        error_list = []

        F_C = solve_banded((1,1),self.ab,self.F_M - r*self.F_N)

        it = 0

        ref_delta_e = None

        exceed_bounds = False
        #while (delta_r > min_delta_r and it < max_iterations) or it < min_iterations:
        while (delta_r > min_delta_r and it < max_iterations):

            F_C = solve_banded((1,1), self.ab, self.F_M - r*self.F_N)

            delta_e = np.mean((self.F_M - F_C - r*self.F_N) * self.F_N)
            r_new = r + learning_rate * delta_e
            
            '''compute error on cross-validation set'''
            F_C_crossval = solve_banded((1,1),self.ab,self.F_M_crossval - r*self.F_N_crossval)

#            error_it = error_calc_outlier(self.F_M_crossval, self.F_N_crossval, F_C_crossval, r)
            error_it = abs(error_calc(self.F_M_crossval, self.F_N_crossval, F_C_crossval, r))
            
            delta_r = np.abs(r_new - r)/r
            r = r_new

            if len(error_list) and error_it > error_list[-1]: # early stopping
                logging.warning("stop: early stopping")
                break

            r_list.append(r)
            error_list.append(error_it)
            it+=1
            
            
            
        # if r or error_it go out of acceptable bounds, break 
        if r < 0.0 or r > 1.0:
            exceed_bounds = True
            logging.warning("stop: r outside of [0.0, 1.0] - (%f)", r)
        if error_it > max_error:
            exceed_bounds = True
            logging.warning("stop: error exceeded bounds")
        if it == max_iterations:
            logging.warning("stop: maximum iterations (%d)", max_iterations)
        if delta_r <= min_delta_r:
            logging.warning("stop: dr < min_dr (%f, %f)", delta_r, min_delta_r)

        F_C = solve_banded((1,1),self.ab,self.F_M - r*self.F_N)

        r_list = np.array(r_list)
        error_list = np.array(error_list)
        self.r_vals = r_list
        self.error_vals = error_list
        self.r = r
        self.F_C = F_C
        self.F_C_crossval = F_C_crossval
        self.it = it

        return exceed_bounds


class NeuropilSubtractGrid (NeuropilSubtract):
    def __init__(self, T, lam=0.01, k=4, steps=3):
        T_k = int(T / k)
        super(NeuropilSubtractGrid, self).__init__(T_k, lam)

        self.k = k
        self.steps = steps

    def set_F(self, F_M, F_N):
        self.F_M = []
        self.F_N = []

        for ki in range(self.k):
            F_M_i = F_M[ki*self.T:(ki+1)*self.T]
            F_N_i = F_N[ki*self.T:(ki+1)*self.T]

            F_N_i_min, F_N_i_max = float(np.amin(F_N_i)), float(np.amax(F_N_i))
            
            # rescale so F_N is [0,1]
            F_M_i_s = (F_M_i - F_N_i_min)/(F_N_i_max - F_N_i_min)
            F_N_i_s = (F_N_i - F_N_i_min)/(F_N_i_max - F_N_i_min)

            self.F_M.append(F_M_i_s)
            self.F_N.append(F_N_i_s)

    def fit(self, r_range=[0.001, 1.999], n=20):
        global_min_error = None
        global_min_r = None

        r_vals = []
        error_vals = []

        step_range = r_range
        for step in range(self.steps):
            logging.debug("step %d", step)
            step_errors = []
            rs = np.linspace(step_range[0], step_range[1], n)
            for r in rs:
                logging.debug("  r %f", r)
                error = self.estimate_error(r)
                step_errors.append(error)

                r_vals.append(r)
                error_vals.append(error)

            min_i = np.argmin(step_errors)
            min_error = step_errors[min_i]
            
            if global_min_error is None or min_error < global_min_error:
                global_min_error = min_error
                global_min_r = rs[min_i]
                
            step_range = [ rs[max(min_i - 1, 0)], rs[min(min_i + 1, len(rs)-1)] ]
            
        self.r_vals = r_vals
        self.error_vals = error_vals
        self.r = global_min_r
        self.error = global_min_error

    def estimate_error(self, r):
        errors = np.zeros(self.k)
        for ki in range(self.k):
            F_M = self.F_M[ki]
            F_N = self.F_N[ki]
            F_C = solve_banded((1,1), self.ab, F_M - r*F_N)
            errors[ki] = abs(error_calc(F_M, F_N, F_C, r))

        return np.mean(errors)
            
def estimate_contamination_ratios_grid(F_M_unscaled, F_N_unscaled, 
                                       lam=0.05, k=4, steps=3,
                                       r_range=[0.001, 1.999], n=20, 
                                       max_error=0.2):
    ns = NeuropilSubtractGrid(len(F_M_unscaled), lam=lam, k=k, steps=steps)
    ns.set_F(F_M_unscaled, F_N_unscaled)
    ns.fit(r_range, n)

    bounds_error = ns.error < 0 or ns.error > max_error

    results = {}
    results["r"] = ns.r
    results["r_vals"] = ns.r_vals
    results["err"] = ns.error_vals[-1]
    results["err_vals"] = ns.error_vals
    results["min_error"] = ns.error
    results["bounds_error"] = bounds_error
    results["it"] = len(ns.r_vals)
    return results


def estimate_contamination_ratios(F_M_unscaled, F_N_unscaled, 
                                  r_init=0.001, learning_rate=10.0, 
                                  lam=0.05):
    ''' Calculates neuropil contamination of ROI

    Parameters
    ----------
    F_M_unscaled: ROI trace

    F_N_unscaled: Neuropil trace

    Returns
    -------
    dictionary: key-value pairs
        * 'r': the contamination ratio -- corrected trace = M - r*N
        * 'err': RMS error
        * 'min_error': minimum error
        * 'bounds_error': boolean. True if error or R are outside tolerance
    '''
    

    T = len(F_M_unscaled)
    assert T == len(F_N_unscaled), "Input arrays of different dimension"
    T_cross_val = int(T/2)
    if T - T_cross_val > T_cross_val:
        T = T - 1
    ns = NeuropilSubtract(T_cross_val, lam=lam)
    F_M_unscaled_cross_val = np.copy(F_M_unscaled)
    F_N_unscaled_cross_val = np.copy(F_N_unscaled)

    '''pick r on first half of the trace '''
    F_M_unscaled = F_M_unscaled[0:T_cross_val]
    F_N_unscaled = F_N_unscaled[0:T_cross_val]        
    
    F_M_unscaled_cross_val = F_M_unscaled_cross_val[T_cross_val:T]
    F_N_unscaled_cross_val =  F_N_unscaled_cross_val[T_cross_val:T]
    
    ''' normalize to have F_N in (0,1)'''
    F_M = (F_M_unscaled - float(np.amin(F_N_unscaled)))/float(np.amax(F_N_unscaled)-np.amin(F_N_unscaled))
    F_N = (F_N_unscaled - float(np.amin(F_N_unscaled)))/float(np.amax(F_N_unscaled)-np.amin(F_N_unscaled))
    F_M_cross_val = (F_M_unscaled_cross_val - float(np.amin(F_N_unscaled_cross_val)))/float(np.amax(F_N_unscaled_cross_val)-np.amin(F_N_unscaled_cross_val))
    F_N_cross_val = (F_N_unscaled_cross_val - float(np.amin(F_N_unscaled_cross_val)))/float(np.amax(F_N_unscaled_cross_val)-np.amin(F_N_unscaled_cross_val))

    # fitting model 
    
    ns.set_F(F_M, F_N, F_M_cross_val, F_N_cross_val)

    '''stop gradient descent at first increase of cross-validation error'''
    #bounds_err = ns.fit_grad_descent(r_init=r_init, 
    # learning_rate=learning_rate)
    #bounds_err = ns.fit_block_descent(r_init=r_init)
    bounds_err = ns.fit_grad_desc_adaptive_step(r_init=r_init, 
                                                learning_rate=learning_rate)
    F_C_unscaled = ns.F_C*float(np.amax(F_N_unscaled)-np.amin(F_N_unscaled)) + (1-ns.r)*float(np.amin(F_N_unscaled))
    F_C_unscaled_crossval = ns.F_C_crossval*float(np.amax(F_N_unscaled_cross_val)-np.amin(F_N_unscaled_cross_val)) + (1-ns.r)*float(np.amin(F_N_unscaled_cross_val))

    min_error = np.zeros(T)
    min_error[:T-T_cross_val] = F_C_unscaled
    min_error[T-T_cross_val:T] = F_C_unscaled_crossval        
    results = {}
    results["r"] = ns.r
    results["r_vals"] = ns.r_vals
    results["err"] = abs(ns.error_vals[-1])
    results["min_error"] = min_error
    results["bounds_error"] = bounds_err
    results["it"] = ns.it
    return results

