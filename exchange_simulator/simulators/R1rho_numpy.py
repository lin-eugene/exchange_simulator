import sympy
import numpy as np
from scipy.linalg import expm
from collections import defaultdict
import matplotlib.pyplot as plt
from simulators.liouvillian import LiouvillianNumPy, LiouvillianSymPy
from scipy.optimize import curve_fit


def exp_decay(t, I0, R1rho):
    return I0 * np.exp(-R1rho*t)

class R1rhoRotationMatrixNumPy:
    def __init__(self, params):
        omega1 = params['omega1'] * 2*np.pi
        omegaRF = params['omegaRF'] * 2*np.pi
        omegaA = params['omegaA'] * 2*np.pi
        omegaB = params['omegaB'] * 2*np.pi
        sin_theta_A = omega1 / np.sqrt(omega1**2 + (omegaA - omegaRF)**2)
        cos_theta_A = (omegaA - omegaRF) / np.sqrt(omega1**2 + (omegaA - omegaRF)**2)
        sin_theta_B = omega1 / np.sqrt(omega1**2 + (omegaB - omegaRF)**2)
        cos_theta_B = (omegaB - omegaRF) / np.sqrt(omega1**2 + (omegaB - omegaRF)**2)
        self.R = np.array(
            [
                [cos_theta_A, 0, -sin_theta_A, 0, 0, 0],
                [0 , 1, 0, 0, 0, 0],
                [sin_theta_A, 0, cos_theta_A, 0, 0, 0],
                [0, 0, 0, cos_theta_B, 0, -sin_theta_B],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, sin_theta_B, 0, cos_theta_B]
            ]
        )

class SimulateR1:
    def __init__(self, params, T_relax):
        self.params = params

        self.T_relax = T_relax
        R1_params = params.copy()
        R1_params['omega1'] = 0
        R1_params['omegaRF'] = 0
        R1_params['omegaA'] = 0

        self.R1_liouvillian = LiouvillianNumPy(R1_params)

    def simulate(self):
        R1_tau = self.T_relax
        eig_val_R1_L, CoB_mat_R1_L = np.linalg.eig(self.R1_liouvillian.L6x6)
        eig_val_R1_L_new = []
        for val in eig_val_R1_L:
            if np.abs(val.imag) > 10**-3:
                eig_val_R1_L_new.append(val*10**9*R1_tau)
            else:
                eig_val_R1_L_new.append(val*R1_tau)

        x_0 = np.array([0,
                        0,
                        1-self.params['pb'],
                        0,
                        0,
                        self.params['pb']])
        
        diag_mat_R1_L = np.diag(np.exp(eig_val_R1_L_new))

        expm_R1_L = CoB_mat_R1_L @ diag_mat_R1_L @ np.linalg.inv(CoB_mat_R1_L)
        self.total_evol = expm_R1_L @ x_0

def calculate_R1(params, taus: list):
    Is = []
    for tau in taus:
        simulator = SimulateR1(params, T_relax=tau)
        simulator.simulate()
        Is.append(simulator.total_evol[2])
    Is = np.array(Is)
    # Is = Is * I0
    # print(Is)
    return Is


class SimulateR1rhoNumPy:
    def __init__(self, params, T_relax):
        
        self.params = params

        self.T_relax = T_relax

        self.R1rho_liouvillian = LiouvillianNumPy(params)

        self.R = R1rhoRotationMatrixNumPy(params)
        R1_params = params.copy()
        R1_params['omega1'] = 0
        R1_params['omegaRF'] = 0
        R1_params['omegaA'] = 0

        self.R1_liouvillian = LiouvillianNumPy(R1_params)

        self.simulate()

    def simulate(self):
        R1_tau = (self.T_relax - self.params['tau']) / 2
        eig_val_R1_L, CoB_mat_R1_L = np.linalg.eig(self.R1_liouvillian.L6x6)
        # print(CoB_mat_R1_L.shape)
        eig_val_R1rho_L, CoB_mat_R1rho_L = np.linalg.eig(self.R1rho_liouvillian.L6x6)
        eig_val_R1_L_new = []
        for val in eig_val_R1_L:
            if np.abs(val.imag) > 10**-3:
                eig_val_R1_L_new.append(val*10**9*R1_tau)
            else:
                eig_val_R1_L_new.append(val*R1_tau)
        eig_val_R1rho_L_new = []
        for val in eig_val_R1rho_L:
            if np.abs(val.imag) > 10**-3:
                eig_val_R1rho_L_new.append(val*10**9*self.params['tau'])
            else:
                eig_val_R1rho_L_new.append(val*self.params['tau'])

        
        x_0 = np.array([0,
                        0,
                        1-self.params['pb'],
                        0,
                        0,
                        self.params['pb']])
        # print(x_0)

        diag_mat_R1_L = np.diag(np.exp(eig_val_R1_L_new))
        diag_mat_R1rho_L = np.diag(np.exp(eig_val_R1rho_L_new))
        # print(x_0)

        expm_R1_L = CoB_mat_R1_L @ diag_mat_R1_L @ np.linalg.inv(CoB_mat_R1_L)
        
        expm_R1rho_L = CoB_mat_R1rho_L @ diag_mat_R1rho_L @ np.linalg.inv(CoB_mat_R1rho_L)
        # print( expm_R1_L @ x_0)
        # print(self.R.R)
        self.total_evol = expm_R1_L @ ((self.R.R) @ expm_R1rho_L @ np.linalg.inv(self.R.R)) @ expm_R1_L @ x_0
        # self.total_evol = expm_R1_L  @ expm_R1rho_L  @ expm_R1_L @ x_0

    

def calculate_Reff(R1rhos, omega_RFs, params):
    omega = (1-params['pb']) * params['omegaA'] + params['pb'] * params['omegaB']
    # print(omega)
    offsets = np.array(omega_RFs)-omega
    # print(offsets)

    omega1Hz = params['omega1']
    # print(f'{omega1Hz=}')
    sin2theta = omega1Hz**2/(omega1Hz**2+offsets**2)
    Reffs = R1rhos/sin2theta
    return Reffs


class CalculateR1rhoProfile:
    def __init__(self, param_dict, omega_RFs, T_relax=0.16):
        self.param_dict, self.omega_RFs, self.T_relax = param_dict, omega_RFs, T_relax
        # print(f'{self.param_dict["omegaA"]=}')

        self.calc_r1rho_numpy()

    def calc_r1rho_numpy(self):
        all = defaultdict(list)
        for omegaRF in self.omega_RFs:
            taus = np.linspace(0.05,0.15,4)
            Is = []
            for tau in taus:
                self.param_dict['tau'] = tau
                self.param_dict['omegaRF'] = omegaRF

                simulator = SimulateR1rhoNumPy(self.param_dict, T_relax=self.T_relax)
                # print(simulator.total_evol)
                Is.append(simulator.total_evol[2] )
            all[omegaRF].append(Is)

        self.param_dict['tau'] = 0
        ref_simulator = SimulateR1rhoNumPy(self.param_dict, T_relax=self.T_relax)
        ref_I = ref_simulator.total_evol[2]

        self.R1rhos = defaultdict(float)
        self.cfasts = defaultdict(float)
        for key, val in all.items():

            popt, pcov = curve_fit(exp_decay, taus, val[0], p0=(1,10))
            self.R1rhos[key] = popt[1]
            self.cfasts[key] = ref_I-popt[0]

        # print(self.omega_RFs)
        self.Reffs = calculate_Reff(np.array(list(self.R1rhos.values())),
                                    self.omega_RFs,
                                    self.param_dict)
                                    
        

# def calc_r1rho_profile_numpy(param_dict, omega_RFs, T_relax=0.15):
#     all = defaultdict(list)
#     for omegaRF in omega_RFs:
#         # omegaRF = omegaRF * 2 * np.pi
#         taus = np.linspace(0.025,0.15,4)
#         Is = []
#         for tau in taus:
#             param_dict['tau'] = tau
#             param_dict['omegaRF'] = omegaRF

#             simulator = SimulateR1rhoNumPy(param_dict, T_relax=T_relax)
#             # print(simulator.total_evol)
#             Is.append(simulator.total_evol[2])
#         all[omegaRF].append(Is)

#     R1rhos = defaultdict(float)
#     for key, val in all.items():

#         popt, pcov = curve_fit(exp_decay, taus, val[0], p0=(1,10))
#         R1rhos[key] = popt[1]
    
#     return R1rhos