from .CEST_models import calculate_CEST_profile, calculate_deltaR2
from .R1rho_numpy import CalculateR1rhoProfile, calculate_R1
from .liouvillian import LiouvillianNumPy
from .cpmg_baldwin import calculate_cpmg_profile
import numpy as np
from .simulate_spectrum import simulate_spectrum as sim_spec
from .unit_conversions import *
from .exsimExchange import spec2Dproj
from .CPMG_numerical import CalculateCPMGProfile

class SimulateExchange:
    def __init__(self, params):
        self.params = params

        assert "R2A" in params.keys()
        assert "R2B" in params.keys()
        assert "R1A" in params.keys()
        assert "R1B" in params.keys()
        assert "pb" in params.keys()
        assert "kex" in params.keys()
        assert "omegaA" in params.keys()
        assert "omegaB" in params.keys()
        # assert "M_eqA" in params.keys()
        # assert "M_eqB" in params.keys()
        assert "B0" in params.keys()

        self.params['M_eqA'] = (1- self.params['pb'])/10
        self.params['M_eqB'] = self.params['pb']/10

        self.simulate_deltaR2()
        self.simulate_intensities()

    def simulate_CEST(self, 
                      omegaRFs: np.ndarray, 
                      omega1: float,
                      time_CEST:float):
        CEST_params = self.params.copy()
        CEST_params['omega1'] = omega1
        self.i_i0 = calculate_CEST_profile(CEST_params, omegaRFs, time_CEST)
        return self.i_i0
    
    def simulate_deltaR2(self):
        R2_params = self.params.copy()
        R2_params['M_eqA'] = 0
        R2_params['M_eqB'] = 0
        self.deltaR2 = calculate_deltaR2(R2_params, timeT2=0.05)
        return self.deltaR2

    def simulate_R1rho(self, omegaRFs:list,
                            T_relax: float=0.16,
                            omega1: float=2000):
        # omega_RFs_rads = [convert_Hz_to_rads(omegaRF) for omegaRF in omegaRFs]
        R1rho_params = self.params.copy()
        R1rho_params['omega1'] = omega1


        self.R1rhos = CalculateR1rhoProfile(R1rho_params, omegaRFs, T_relax=T_relax)

        return self.R1rhos
    
    def simulate_R1rho_inhomogeneous_B1(self, omegaRFs:list,
                            T_relax: float=0.16,
                            omega1: float=2000):
        R1rho_params = self.params.copy()
        omega1_values = [0.95, 1.0, 1.05]
        omega1_weights = [0.25, 0.5, 0.25]

        R1rhos = []
        cfasts = []
        Reffs = []

        for omega1_val, weight in zip(omega1_values, omega1_weights):
            R1rho_params['omega1'] = omega1 * omega1_val
            R1rho_profile = CalculateR1rhoProfile(R1rho_params, omegaRFs, T_relax=T_relax)
            R1rhos.append(np.array(list(R1rho_profile.R1rhos.values())) * weight)
            cfasts.append(np.array(list(R1rho_profile.cfasts.values())) * weight)
            Reffs.append(np.array(list(R1rho_profile.Reffs)) * weight)
        
        self.R1rhos_inh = np.sum(R1rhos, axis=0)
        self.cfasts_inh = np.sum(cfasts, axis=0)
        self.Reffs_inh = np.sum(Reffs, axis=0)
        return self.R1rhos_inh


    def simulate_CPMG(self, nu_cpmgs: list, 
                      T_relax: float):
        kex = self.params['kex']
        pb = self.params['pb']
        d_omega = np.abs(self.params['omegaA'] - self.params['omegaB']) * 2 * np.pi
        # print(d_omega)
        R2A = self.params['R2A']
        R2B = self.params['R2B']

        self.I_cpmg = calculate_cpmg_profile(kex, pb, d_omega, nu_cpmgs, T_relax, R2A, R2B)
        return self.I_cpmg
    
    def simulate_CPMG_numerical(self, ncycs: list[int], omega1: float, T_relax: float):

        cpmg_params = self.params.copy()
        cpmg_params['omega1'] = omega1
        cpmg_params['omegaRF'] = 0
        self.cpmg_numerical = CalculateCPMGProfile(cpmg_params, ncycs, T_relax=T_relax)

        return self.cpmg_numerical.nu_cpmgs, self.cpmg_numerical.R2effs

    
    def simulate_R1(self, taus: list):
        self.R1s = calculate_R1(self.params, taus)
        return self.R1s

    def simulate_spectrum(self, omega_RF):
        self.params['omegaRF'] = omega_RF
        self.params['omega1'] = 0

        liouvillian = LiouvillianNumPy(self.params).L2x2
        self.spectrum = sim_spec(liouvillian_matrix=liouvillian,
                            params=self.params)
        return self.spectrum
    
    def simulate_intensities(self,):
        kex = self.params['kex']
        pb = self.params['pb']
        dwN = 2*np.pi*(self.params['omegaB'] - self.params['omegaA'])
        dwH = 0.01# 2*np.pi*(self.params['omegaB_H'] -  self.params['omegaA_H'])
        R2g = self.params['R2A']
        R2e = self.params['R2B']
        dfrq = self.params['B0']
        sfrq = self.params['B0'] * 9.8650320809
        self.hfreq,self.nfreq,self.intensity = spec2Dproj(sfrq,dfrq,kex,pb,dwH,dwN,R2g,R2e)
        return self.intensity

        
        