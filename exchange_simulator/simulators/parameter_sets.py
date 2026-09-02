import numpy as np

def dest_parameters(R2A,
                R2B,
                R1A,
                R1B,
                pb,
                kex,
                omegaA,
                omegaB,
                omegaRF,
                omega1,
                ThetaA,
                ThetaB,):
                
    return {'R2A': R2A,
            'R2B': R2B,
            'R1A': R1A,
            'R1B': R1B,
            'pb': pb,
            'kex': kex,
            'omegaA': omegaA,
            'omegaB': omegaB,
            'omegaRF': omegaRF,
            'omega1': omega1,
            'ThetaA': ThetaA,
            'ThetaB': ThetaB}

def deltaR2_parameters(R2A,
                R2B,
                R1A,
                R1B,
                pb,
                kex,
                ThetaA,
                ThetaB,):
                
    return {'R2A': R2A ,
            'R2B': R2B ,
            'R1A': R1A,
            'R1B': R1B,
            'pb': pb,
            'kex': kex,
            'omegaA': 0,
            'omegaB': 0,
            'omegaRF': 0,
            'omega1': 0,
            'ThetaA': ThetaA,
            'ThetaB': ThetaB}
