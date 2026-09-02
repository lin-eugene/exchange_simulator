import sympy
import numpy as np
from scipy.linalg import expm
from collections import defaultdict
import matplotlib.pyplot as plt
from liouvillian import LiouvillianNumPy, LiouvillianSymPy

class Liouvillian:
    def __init__(self):
        super(Liouvillian, self).__init__()
        self.R2A,self.R2B = sympy.symbols('R_2^A R_2^B')
        self.R1A,self.R1B = sympy.symbols('R_1^A R_1^B')
        self.pb = sympy.symbols('p_b')
        self.kex = sympy.symbols('k_{ex}')
        self.omegaA,self.omegaB,self.omegaRF = sympy.symbols('\omega_A \omega_B \omega_{RF}')
        self.omega1 = sympy.symbols('\omega_1')
    
        self.L = sympy.Matrix(
            [
            [-self.R2A-self.pb*self.kex,     self.omegaA-self.omegaRF,    0, (1-self.pb)*self.kex, 0,            0],
            [-(self.omegaA-self.omegaRF),       -self.R2A-self.pb*self.kex, self.omega1,   0,      (1-self.pb)*self.kex, 0],
            [ 0,           -self.omega1,   (-self.R1A-self.pb*self.kex), 0,      0,    (1-self.pb)*self.kex],
            [(self.pb)*self.kex,     0,             0,      (-self.R2B-(1-self.pb)*self.kex), self.omegaB-self.omegaRF,  0],
            [0,       (self.pb)*self.kex,           0, -(self.omegaB-self.omegaRF), (-self.R2B-(1-self.pb)*self.kex), self.omega1],
            [0,            0,      (self.pb)*self.kex, 0,      -self.omega1, (-self.R1B-(1-self.pb)*self.kex)]
        ]
        )

class R1rhoRotationMatrix:
    def __init__(self):
        super(R1rhoRotationMatrix, self).__init__()
        self.omega1 = sympy.symbols('\omega_1')
        self.omegaRF = sympy.symbols('\omega_{RF}')
        self.omegaA,self.omegaB = sympy.symbols('\omega_A \omega_B')
        sin_theta_A = self.omega1 / sympy.sqrt(self.omega1**2 + (self.omegaA - self.omegaRF)**2)
        cos_theta_A = (self.omegaA - self.omegaRF) / sympy.sqrt(self.omega1**2 + (self.omegaA - self.omegaRF)**2)
        sin_theta_B = self.omega1 / sympy.sqrt(self.omega1**2 + (self.omegaB - self.omegaRF)**2)
        cos_theta_B = (self.omegaB - self.omegaRF) / sympy.sqrt(self.omega1**2 + (self.omegaB - self.omegaRF)**2)
        self.R = sympy.Matrix(
            [
                [cos_theta_A, 0, -sin_theta_A, 0, 0, 0],
                [0 , 1, 0, 0, 0, 0],
                [sin_theta_A, 0, cos_theta_A, 0, 0, 0],
                [0, 0, 0, cos_theta_B, 0, -sin_theta_B],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, sin_theta_B, 0, cos_theta_B]
            ]
        )


class SimulateCTR1rho(Liouvillian, R1rhoRotationMatrix):
    def __init__(self):
        super(SimulateCTR1rho, self).__init__()



# def setup_R1rho_mat(R1_liouvillian, R1rho_liouvillian):
#     T_relax_symbol = sympy.symbols('T_{relax}')
#     tau_symbol = sympy.symbols('\tau')
#     R1_time = (T_relax_symbol-tau_symbol)/2
#     R1_mat = R1_liouvillian.L * R1_time
#     R1_mat = R1_mat.exp()

#     R1rho_mat = R1rho_liouvillian.L * tau_symbol
#     R1rho_mat = R1rho_mat.exp()

#     R1_rho_evol = R1rho_liouvillian.R.inv() * R1rho_mat * R1rho_liouvillian.R
#     total_evol = R1_mat * R1_rho_evol * R1_mat

#     return total_evol





def simulate_R1rho(param_dict):
    tau = param_dict['tau']
    omegaRF = param_dict['omegaRF']
    T_relax = param_dict['T_relax']
    R1_tau = (T_relax - tau) / 2
    omega1 = param_dict['omega1']
    omegaA = param_dict['omegaA']
    omegaB = param_dict['omegaB']
    pb = param_dict['pb']
    kex = param_dict['kex']
    R1A = param_dict['R1A']
    R1B = param_dict['R1B']
    R2A = param_dict['R2A']
    R2B = param_dict['R2B']

    R1rho = SimulateCTR1rho()
    R1 = Liouvillian()
    R1rho.L = R1rho.L.subs({
        R1rho.omega1: omega1,
        R1rho.omegaRF: omegaRF,
        R1rho.omegaA: omegaA,
        R1rho.omegaB: omegaB,
        R1rho.pb: pb,
        R1rho.kex: kex,
        R1rho.R1A: R1A,
        R1rho.R1B: R1B,
        R1rho.R2A: R2A,
        R1rho.R2B: R2B
    })
    R1rho.R = R1rho.R.subs({
        R1rho.omega1: omega1,
        R1rho.omegaRF: omegaRF,
        R1rho.omegaA: omegaA,
        R1rho.omegaB: omegaB
    })
    # R1.L = R1.L.subs({
    #     R1.omega1: 0,
    #     R1.omegaRF: omegaRF,
    #     R1.omegaA: omegaA,
    #     R1.omegaB: omegaB,
    #     R1.pb: pb,
    #     R1.kex: kex,
    #     R1.R1A: R1A,
    #     R1.R1B: R1B,
    #     R1.R2A: R2A,
    #     R1.R2B: R2B
    # })

    R1.L = R1.L.subs({
        R1.omega1: 0,
        R1.omegaRF: 0,
        R1.omegaA: 0,
        R1.omegaB: 0,
        R1.pb: pb,
        R1.kex: 0,
        R1.R1A: R1A,
        R1.R1B: R1B,
        R1.R2A: 0,
        R1.R2B: 0
    })
    # print(tau)
    x_0 = np.array([0,0,1-pb,0,0,pb])

    R = sympy.matrix2numpy(R1rho.R, dtype=float)
    R1rho_L = sympy.matrix2numpy(R1rho.L, dtype=float)
    # # print(R1rho_L)
    R1_L =sympy.matrix2numpy(R1.L, dtype=float)
    # # # print(R1_L)
    # # evol = expm(R1_L*R1_tau) @ x_0
    # # print(f'{evol=}')
    # total_evol = expm(-R1_L*R1_tau) @ (np.linalg.inv(R) @ expm(-R1rho_L*tau) @ R) @ expm(-R1_L*R1_tau) @ x_0
    # print(total_evol)

    eig_val_R1_L, CoB_mat_R1_L = np.linalg.eig(R1_L)
    eig_val_R1rho_L, CoB_mat_R1rho_L = np.linalg.eig(R1rho_L)
    eig_val_R1_L_new = []
    for val in eig_val_R1_L:
        if np.abs(val.imag) > 10**-3:
            eig_val_R1_L_new.append(val*10**9*R1_tau)
        else:
            eig_val_R1_L_new.append(val*R1_tau)
    eig_val_R1rho_L_new = []
    for val in eig_val_R1rho_L:
        if np.abs(val.imag) > 10**-3:
            eig_val_R1rho_L_new.append(val*10**9*tau)
        else:
            eig_val_R1rho_L_new.append(val*tau)
    
    diag_mat_R1_L = np.diag(np.exp(eig_val_R1_L_new))
    diag_mat_R1rho_L = np.diag(np.exp(eig_val_R1rho_L_new))

    expm_R1_L = CoB_mat_R1_L @ diag_mat_R1_L @ np.linalg.inv(CoB_mat_R1_L)
    expm_R1rho_L = CoB_mat_R1rho_L @ diag_mat_R1rho_L @ np.linalg.inv(CoB_mat_R1rho_L)

    # evol = expm_R1_L @ x_0
    # print(f'{evol=}')

    total_evol = expm_R1_L @ (np.linalg.inv(R) @ expm_R1rho_L @ R) @ expm_R1_L @ x_0


    return total_evol



class R1rhoRotationMatrixNumPy:
    def __init__(self, params):
        omega1 = params['omega1']
        omegaRF = params['omegaRF']
        omegaA = params['omegaA']
        omegaB = params['omegaB']
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

# class SimulateCTR1rhoNumPy(LiouvillianNumPy, R1rhoRotationMatrixNumPy):
#     def __init__(self, params):
#         super(SimulateCTR1rhoNumPy, self).__init__(params)


class SimulateR1rhoNumPy():
    def __init__(self, params):
        R1_params = params.copy()
        R1_params['omega1'] = 0
        R1_params['omegaRF'] = 0
        R1_params['omegaA'] = 0
        R1_params['omegaB'] = 0
        R1_params['kex'] = 0
        R1_params['R2A'] = 0
        R1_params['R2B'] = 0

        self.R1rho_liouvillian = LiouvillianNumPy(params)
        self.R1rho_rotation = R1rhoRotationMatrixNumPy(params)
        self.R1_liouvillian = LiouvillianNumPy(R1_params)

        self.params = params
    
    def simulate(self):
        R1_tau = (self.params['T_relax'] - self.params['tau'])
        eig_val_R1_L, CoB_mat_R1_L = np.linalg.eig(self.R1_liouvillian.L)
        eig_val_R1rho_L, CoB_mat_R1rho_L = np.linalg.eig(self.R1rho_liouvillian.L)
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

        x_0 = np.array([0,0,1-pb,0,0,pb])

        diag_mat_R1_L = np.diag(np.exp(eig_val_R1_L_new))
        diag_mat_R1rho_L = np.diag(np.exp(eig_val_R1rho_L_new))

        expm_R1_L = CoB_mat_R1_L @ diag_mat_R1_L @ np.linalg.inv(CoB_mat_R1_L)
        expm_R1rho_L = CoB_mat_R1rho_L @ diag_mat_R1rho_L @ np.linalg.inv(CoB_mat_R1rho_L)
        total_evol = expm_R1_L @ (np.linalg.inv(R) @ expm_R1rho_L @ R) @ expm_R1_L @ x_0


        return total_evol



def simulate_R1rho(param_dict):
    tau = param_dict['tau']
    omegaRF = param_dict['omegaRF']
    T_relax = param_dict['T_relax']
    R1_tau = (T_relax - tau) / 2
    omega1 = param_dict['omega1']
    omegaA = param_dict['omegaA']
    omegaB = param_dict['omegaB']
    pb = param_dict['pb']
    kex = param_dict['kex']
    R1A = param_dict['R1A']
    R1B = param_dict['R1B']
    R2A = param_dict['R2A']
    R2B = param_dict['R2B']



    R1rho = SimulateCTR1rho()
    R1 = Liouvillian()
    R1rho.L = R1rho.L.subs({
        R1rho.omega1: omega1,
        R1rho.omegaRF: omegaRF,
        R1rho.omegaA: omegaA,
        R1rho.omegaB: omegaB,
        R1rho.pb: pb,
        R1rho.kex: kex,
        R1rho.R1A: R1A,
        R1rho.R1B: R1B,
        R1rho.R2A: R2A,
        R1rho.R2B: R2B
    })
    R1rho.R = R1rho.R.subs({
        R1rho.omega1: omega1,
        R1rho.omegaRF: omegaRF,
        R1rho.omegaA: omegaA,
        R1rho.omegaB: omegaB
    })
    # R1.L = R1.L.subs({
    #     R1.omega1: 0,
    #     R1.omegaRF: omegaRF,
    #     R1.omegaA: omegaA,
    #     R1.omegaB: omegaB,
    #     R1.pb: pb,
    #     R1.kex: kex,
    #     R1.R1A: R1A,
    #     R1.R1B: R1B,
    #     R1.R2A: R2A,
    #     R1.R2B: R2B
    # })

    R1.L = R1.L.subs({
        R1.omega1: 0,
        R1.omegaRF: 0,
        R1.omegaA: 0,
        R1.omegaB: 0,
        R1.pb: pb,
        R1.kex: 0,
        R1.R1A: R1A,
        R1.R1B: R1B,
        R1.R2A: 0,
        R1.R2B: 0
    })
    # print(tau)
    x_0 = np.array([0,0,1-pb,0,0,pb])

    R = sympy.matrix2numpy(R1rho.R, dtype=float)
    R1rho_L = sympy.matrix2numpy(R1rho.L, dtype=float)
    # # print(R1rho_L)
    R1_L =sympy.matrix2numpy(R1.L, dtype=float)
    # # # print(R1_L)
    # # evol = expm(R1_L*R1_tau) @ x_0
    # # print(f'{evol=}')
    # total_evol = expm(-R1_L*R1_tau) @ (np.linalg.inv(R) @ expm(-R1rho_L*tau) @ R) @ expm(-R1_L*R1_tau) @ x_0
    # print(total_evol)

    eig_val_R1_L, CoB_mat_R1_L = np.linalg.eig(R1_L)
    eig_val_R1rho_L, CoB_mat_R1rho_L = np.linalg.eig(R1rho_L)
    eig_val_R1_L_new = []
    for val in eig_val_R1_L:
        if np.abs(val.imag) > 10**-3:
            eig_val_R1_L_new.append(val*10**9*R1_tau)
        else:
            eig_val_R1_L_new.append(val*R1_tau)
    eig_val_R1rho_L_new = []
    for val in eig_val_R1rho_L:
        if np.abs(val.imag) > 10**-3:
            eig_val_R1rho_L_new.append(val*10**9*tau)
        else:
            eig_val_R1rho_L_new.append(val*tau)
    
    diag_mat_R1_L = np.diag(np.exp(eig_val_R1_L_new))
    diag_mat_R1rho_L = np.diag(np.exp(eig_val_R1rho_L_new))

    expm_R1_L = CoB_mat_R1_L @ diag_mat_R1_L @ np.linalg.inv(CoB_mat_R1_L)
    expm_R1rho_L = CoB_mat_R1rho_L @ diag_mat_R1rho_L @ np.linalg.inv(CoB_mat_R1rho_L)

    evol = expm_R1_L @ x_0
    # print(f'{evol=}')

    total_evol = expm_R1_L @ (np.linalg.inv(R) @ expm_R1rho_L @ R) @ expm_R1_L @ x_0


    return total_evol


