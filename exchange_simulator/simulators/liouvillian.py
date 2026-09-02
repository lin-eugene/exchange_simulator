import sympy
import numpy as np
from scipy.linalg import expm
from collections import defaultdict
import matplotlib.pyplot as plt

class LiouvillianSymPy:
    def __init__(self):
        super(LiouvillianSymPy, self).__init__()
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

class LiouvillianNumPy:
    def __init__(self, params):
        self.R2A = params['R2A']
        self.R2B = params['R2B']
        self.R1A = params['R1A']
        self.R1B = params['R1B']
        self.pb = params['pb']
        self.kex = params['kex']
        self.omegaA = params['omegaA'] *2*np.pi
        self.omegaB = params['omegaB'] *2*np.pi
        self.omegaRF = params['omegaRF'] *2*np.pi
        self.omega1 = params['omega1'] *2*np.pi

        # print(f'{self.omegaA=}, {self.omegaB=}, {self.omegaRF=}, {self.omega1=}')

        MeqA = params['M_eqA']
        MeqB = params['M_eqB']



        self.L = np.array([[0,0,0,0,0,0,0],
            [0,-self.R2A-self.pb*self.kex,     self.omegaA-self.omegaRF,    0, (1-self.pb)*self.kex, 0,            0],
            [0,-(self.omegaA-self.omegaRF),       -self.R2A-self.pb*self.kex, self.omega1,   0,      (1-self.pb)*self.kex, 0],
            [2*self.R1A*MeqA, 0,           -self.omega1,   (-self.R1A-self.pb*self.kex), 0,      0,    (1-self.pb)*self.kex],
            [0, (self.pb)*self.kex,     0,             0,      (-self.R2B-(1-self.pb)*self.kex), self.omegaB-self.omegaRF,  0],
            [0, 0,       (self.pb)*self.kex,           0, -(self.omegaB-self.omegaRF), (-self.R2B-(1-self.pb)*self.kex), self.omega1],
            [2*self.R1B*MeqB, 0,            0,      (self.pb)*self.kex, 0,      -self.omega1, (-self.R1B-(1-self.pb)*self.kex)]
        ])
            
            
            
            
        #     [
        #     [0, 0, 0, 0, 0, 0, 0],
        #     [0, -1*(-self.R2A-self.pb*self.kex),     self.omegaA-self.omegaRF,    0, -(1-self.pb)*self.kex, 0,            0],
        #     [0, -(self.omegaA-self.omegaRF),       -1*(-self.R2A-self.pb*self.kex), self.omega1,   0,      -(1-self.pb)*self.kex, 0],
        #     [2*self.R1A*MeqA, 0,           -self.omega1,   -1*(-self.R1A-self.pb*self.kex), 0,      0,    -(1-self.pb)*self.kex],
        #     [0, -(self.pb)*self.kex,     0,             0,      -1*(-self.R2B-(1-self.pb)*self.kex), self.omegaB-self.omegaRF,  0],
        #     [0, 0,       -(self.pb)*self.kex,           0, -(self.omegaB-self.omegaRF), -1*(-self.R2B-(1-self.pb)*self.kex), self.omega1],
        #     [2*self.R1B*MeqB, 0,            0,      -(self.pb)*self.kex, 0,      -self.omega1, -1*(-self.R1B-(1-self.pb)*self.kex)]
        # ])
        # self.L= np.array(
        # [   [0, 0, 0, 0, 0, 0, 0],
        #     [0,-self.R2A-self.pb*self.kex,     self.omegaA-self.omegaRF,    0, (1-self.pb)*self.kex, 0,            0],
        #     [0,-(self.omegaA-self.omegaRF),       -self.R2A-self.pb*self.kex, self.omega1,   0,      (1-self.pb)*self.kex, 0],
        #     [-2*self.R1A*MeqA, 0,           -self.omega1,   (-self.R1A-self.pb*self.kex), 0,      0,    (1-self.pb)*self.kex],
        #     [0,(self.pb)*self.kex,     0,             0,      (-self.R2B-(1-self.pb)*self.kex), self.omegaB-self.omegaRF,  0],
        #     [0,0,       (self.pb)*self.kex,           0, -(self.omegaB-self.omegaRF), (-self.R2B-(1-self.pb)*self.kex), self.omega1],
        #     [-2*self.R1B*MeqB,0,            0,      (self.pb)*self.kex, 0,      -self.omega1, (-self.R1B-(1-self.pb)*self.kex)]
        # ]
        # )

        # [[0, 0, 0, 0, 0, 0, 0],
        #     [0,-self.R2A-self.pb*self.kex,     self.omegaA-self.omegaRF,    0, (1-self.pb)*self.kex, 0,            0],
        #     [0,-(self.omegaA-self.omegaRF),       -self.R2A-self.pb*self.kex, self.omega1,   0,      (1-self.pb)*self.kex, 0],
        #     [-2*self.R1A*MeqA, 0,           -self.omega1,   (-self.R1A-self.pb*self.kex), 0,      0,    (1-self.pb)*self.kex],
        #     [0,(self.pb)*self.kex,     0,             0,      (-self.R2B-(1-self.pb)*self.kex), self.omegaB-self.omegaRF,  0],
        #     [0,0,       (self.pb)*self.kex,           0, -(self.omegaB-self.omegaRF), (-self.R2B-(1-self.pb)*self.kex), self.omega1],
        #     [-2*self.R1B*MeqB,0,            0,      (self.pb)*self.kex, 0,      -self.omega1, (-self.R1B-(1-self.pb)*self.kex)]
        # ]

        self.L6x6 = self.L[1:7,1:7]

        self.L2x2 = np.array([[1j*self.omegaA-(self.R2A+self.kex), self.kex],
              [self.kex, 1j*self.omegaB-(self.R2B+self.kex)]])



