from simulators.liouvillian import LiouvillianNumPy
import simulators.parameter_sets
import numpy as np
from itertools import repeat

from multiprocessing import Pool


def _calculate_I_I0(liouvillian: np.ndarray, 
                     c_init: np.ndarray, 
                     time: float):
    # mat_exp = lambda t: expm(liouvillian * time)

    # mat_end = mat_exp(time)
    
    eig_val, CoB_mat = np.linalg.eig(liouvillian)
    eig_val_new = []
    for val in eig_val:
        if np.abs(val.imag) > 10**-3:
            eig_val_new.append(val*10**9 *time)
        else:
            eig_val_new.append(val *time)
    diag_mat = np.diag(np.exp(eig_val_new) )
    expm_approx = CoB_mat @ diag_mat @ np.linalg.inv(CoB_mat)

    c_array = expm_approx @ c_init
    return c_array

def calculate_CEST_profile(params,
                           omegaRFs: list,
                            timeCEST: float):
    pb = params['pb']

    results = []
    for omegaRF in omegaRFs:
        params['omegaRF'] = omegaRF

        liouvillian_matrix = LiouvillianNumPy(params).L

        c_init = np.array([0.5,
                        0,
                        0,
                        1-pb,
                        0,
                        0,
                        pb])
        times = [0, timeCEST]
        c_array = np.array(list(map(_calculate_I_I0,
                                    repeat(liouvillian_matrix),
                                    repeat(c_init),
                                    times)
                                    ))
        
        i_i0 = (c_array[1,3]) / (c_array[0,3])
        results.append(i_i0)

    return results

# def calculate_DEST_profile(x,
#                 R2A,
#                 R2B,
#                 R1A,
#                 R1B,
#                 pb,
#                 kex,
#                 omegaA,
#                 omegaB,
#                 omega1,
#                 ThetaA,
#                 ThetaB,
#                 timeCEST):
#     with Pool() as pool:

#     return _calculate_DEST(R2A,
#                 R2B,
#                 R1A,
#                 R1B,
#                 pb,
#                 kex,
#                 omegaA,
#                 omegaB,
#                 omegaRF,
#                 omega1,
#                 ThetaA,
#                 ThetaB, 
#                 timeCEST)


def calculate_R2obs(
                      params,
                timeT2=0.05):
    
    pb = params['pb']
    params['omegaRF'] = 0
    params['omega1'] = 0
    params['omegaA'] = 0
    params['omegaB'] = 0
    
    liouvillian_matrix = LiouvillianNumPy(params).L
    c_init = np.array([0.5,
                       1-pb,
                       0,
                       0,
                       pb,
                       0,
                       0])
    times = [0, timeT2]
    c_array = np.array(list(map(_calculate_I_I0,
                                repeat(liouvillian_matrix),
                                repeat(c_init),
                                times)
                                ))
    component = 1 # IxA
    i_i0 = c_array[1,component] / c_array[0,component]
    R2_observed = -np.log(i_i0) / timeT2

    return R2_observed

def calculate_deltaR2(params,
                timeT2=0.05):
    R2_obs = calculate_R2obs(params, timeT2=timeT2)
    params_no_excahange = params.copy()
    params_no_excahange['kex'] = 0
    params_no_excahange['pb'] = 0
    R2_no_exchange = calculate_R2obs(params_no_excahange, timeT2=timeT2)
    delta_R2 = R2_obs - R2_no_exchange
    return delta_R2




