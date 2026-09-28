from .liouvillian import LiouvillianNumPy
import numpy as np
from scipy.linalg import expm
from collections import defaultdict
from .compute_fid import calculate_fid, calculate_max_intensity, compute_spectrum

from numpy.linalg import matrix_power

class RFLiouvillian:
    def __init__(self, params):

        self.omega1 = params['omega1'] *2*np.pi
        params['phi']= 0 
        self.phi = params['phi']

        # print(f'{self.omegaA=}, {self.omegaB=}, {self.omegaRF=}, {self.omega1=}')

        self.L = np.array([[0,0,0,0,0,0,0],
            [0,0,       0, self.omega1*np.sin(self.phi), 0, 0,            0],
            [0,0,      0, -self.omega1*np.cos(self.phi),   0,      0, 0],
            [0, self.omega1*np.sin(self.phi), self.omega1*np.cos(self.phi),   0, 0,      0,    0],
            [0, 0,     0,             0,      0, 0,  self.omega1*np.sin(self.phi)],
            [0, 0,       0,           0, 0, 0, -self.omega1*np.cos(self.phi)],
            [0, 0,    0,      0, self.omega1*np.sin(self.phi),      self.omega1*np.cos(self.phi), 0]
        ])


class SimulateCPMG:
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

        self.rf_liouvillian = RFLiouvillian(params).L[1:7, 1:7]
        # print(self.rf_liouvillian)
        self.pi_duration = 1/(2*params['omega1'])
        # print(self.pi_duration)

        free_precession_liouvillian_params = params.copy()
        free_precession_liouvillian_params['omega1'] = 0
        self.free_precession_liouvillian = LiouvillianNumPy(free_precession_liouvillian_params).L6x6

        
    # def _detect(self, mag_vec):

    #     delta_omega = 2*np.pi*(self.params['omegaA'] - self.params['omegaB'])
    #     delta_R2 = self.params['R2B'] - self.params['R2A']
    #     k_ge = self.params['pb'] * self.params['kex']
    #     k_eg = (1 - self.params['pb']) * self.params['kex']

    #     h1 = 2*delta_omega*(delta_R2+k_eg-k_ge)
    #     h2 = (delta_R2+k_eg-k_ge)**2+4*k_eg*k_ge-delta_omega**2

    #     h3 = 1/np.sqrt(2) * np.sqrt(h2+np.sqrt(h1**2+h2**2))
    #     h4 = 1/np.sqrt(2) * np.sqrt(-h2+np.sqrt(h1**2+h2**2))

    #     f00 = 1/2 * (delta_R2+self.params['kex']-h3) + np.imag/2 * (delta_omega-h4)
    #     f11 = 1/2 * (delta_R2+self.params['kex']+h3) + np.imag/2 * (delta_omega+h4)



    def simulate_CPMG(self,
                      ncyc: np.ndarray,
                      T_relax: float=0.08):

        self.T_relax = T_relax
        self.nu_cpmg = ncyc / self.T_relax
        # nu_cpmg=ncyc/T_relax
        self.tau_cpmg = self.T_relax/(4.0*ncyc) 

        # print(self.nu_cpmg, self.tau_cpmg)

        L = self.rf_liouvillian #+ self.free_precession_liouvillian
        pulse_propagator = expm(L * self.pi_duration)
        tau_propagator = expm(self.free_precession_liouvillian * self.tau_cpmg)
        # eig_val_pulse, CoB_mat_pulse = np.linalg.eig(L * self.pi_duration)
        # eig_val_delay, CoB_mat_delay = np.linalg.eig(self.free_precession_liouvillian * self.tau_cpmg)

        # diag_mat_pulse = np.diag(np.exp(eig_val_pulse))
        # diag_mat_delay = np.diag(np.exp(eig_val_delay))

        # pulse_propagator = CoB_mat_pulse @ diag_mat_pulse @ np.linalg.inv(CoB_mat_pulse)
        # tau_propagator = CoB_mat_delay @ diag_mat_delay @ np.linalg.inv(CoB_mat_delay)
        

        x_0 = np.array([
                        # 0,
                        0,
                        1-self.params['pb'],
                        0,
                        0,
                        self.params['pb'],
                        0])
        # print(f'{x_0=}')
        # print((tau_propagator @ pulse_propagator @ tau_propagator @ tau_propagator @ pulse_propagator @ tau_propagator) @ x_0)
        cpmg = (tau_propagator @ pulse_propagator @ tau_propagator @ tau_propagator @ pulse_propagator @ tau_propagator)
        total_propagator = matrix_power(cpmg, ncyc)
        
        self.total_evol = total_propagator @ x_0
        # print(self.total_evol)
        # print(self.free_precession_liouvillian)
        self.fid = calculate_fid(self.free_precession_liouvillian,
                            self.total_evol,
                            dwell_time=1e-4,
                            duration=0.1)
        self.spectrum = compute_spectrum(self.fid)
        
        self.fid_I0 = calculate_fid(self.free_precession_liouvillian,
                            x_0,
                            dwell_time=1e-4,
                            duration=0.1)
        self.spectrum_I0 = compute_spectrum(self.fid_I0)
        # print(self.spectrum)
        # print(total_evol[1]/x_0[1])

        I = calculate_max_intensity(self.spectrum)  
        I0  = calculate_max_intensity(self.spectrum_I0) 

        # I_sum = np.sum(np.abs(self.spectrum['intensity']))
        # I0_sum  = np.sum(np.abs(self.spectrum_I0['intensity']))
        R2eff = -1/self.T_relax * np.log(I/I0)
        # R2eff = -1/self.T_relax * np.log(I_sum/I0_sum)

       # R2eff = -1/self.T_relax * np.log(self.total_evol[1]/x_0[1])

        return R2eff#, R2eff_numerical, R2eff_sum


class CalculateCPMGProfile:
    def __init__(self, params, ncycs, T_relax: float=0.08):
        self.params = params
        self.ncycs = ncycs
        self.T_relax = T_relax

        self.calc_cpmg_profile()

    def calc_cpmg_profile(self):
        sim_cpmg = SimulateCPMG(self.params)
        self.R2effs = []
        self.nu_cpmgs = []
        for ncyc in self.ncycs:
            R2eff = sim_cpmg.simulate_CPMG(ncyc, T_relax=self.T_relax)
            self.R2effs.append(R2eff)
            self.nu_cpmgs.append(sim_cpmg.nu_cpmg)
        # return self.nu_cpmgs, self.R2effs

        


# class SimulateCPMGDaiwen:
#     def __init__(self, params):
#         self.params = params

#         assert "R2A" in params.keys()
#         assert "R2B" in params.keys()
#         assert "R1A" in params.keys()
#         assert "R1B" in params.keys()
#         assert "pb" in params.keys()
#         assert "kex" in params.keys()
#         assert "omegaA" in params.keys()
#         assert "omegaB" in params.keys()
#         # assert "M_eqA" in params.keys()
#         # assert "M_eqB" in params.keys()
#         assert "B0" in params.keys()


#         self.params['M_eqA'] = (1- self.params['pb'])/10
#         self.params['M_eqB'] = self.params['pb']/10

#         self.rf_liouvillian = RFLiouvillian(params).L[1:7, 1:7]
#         # print(self.rf_liouvillian)
#         self.pi_duration = 1/(2*params['omega1'])
#         # print(self.pi_duration)

#         free_precession_liouvillian_params = params.copy()
#         free_precession_liouvillian_params['omega1'] = 0
#         self.free_precession_liouvillian = LiouvillianNumPy(free_precession_liouvillian_params).L6x6

        
#     def calculate_nm(self, ncyc: int) -> tuple:
#         m = ncyc % 2
#         n = 0
#         for i in range(ncyc):
#             if (i+1) % 2 == 1:
#                 n += 1
#         return n, m

#     def simulate_CPMG(self,
#                       ncyc: np.ndarray,
#                       T_relax: float=0.08):

#         self.T_relax = T_relax
#         self.nu_cpmg = ncyc / self.T_relax
#         # nu_cpmg=ncyc/T_relax
#         self.tau_cpmg = self.T_relax/(4.0*ncyc) 
#         n, m = self.calculate_nm(ncyc)

#         # print(self.nu_cpmg, self.tau_cpmg)

#         L = self.rf_liouvillian #+ self.free_precession_liouvillian
#         pulse_propagator = expm(L * self.pi_duration)
#         tau_propagator = expm(self.free_precession_liouvillian * self.tau_cpmg)
#         # eig_val_pulse, CoB_mat_pulse = np.linalg.eig(L * self.pi_duration)
#         # eig_val_delay, CoB_mat_delay = np.linalg.eig(self.free_precession_liouvillian * self.tau_cpmg)

#         # diag_mat_pulse = np.diag(np.exp(eig_val_pulse))
#         # diag_mat_delay = np.diag(np.exp(eig_val_delay))

#         # pulse_propagator = CoB_mat_pulse @ diag_mat_pulse @ np.linalg.inv(CoB_mat_pulse)
#         # tau_propagator = CoB_mat_delay @ diag_mat_delay @ np.linalg.inv(CoB_mat_delay)
        

#         x_0 = np.array([
#                         # 0,
#                         0,
#                         1-self.params['pb'],
#                         0,
#                         0,
#                         self.params['pb'],
#                         0])
#         # print(f'{x_0=}')
#         # print((tau_propagator @ pulse_propagator @ tau_propagator @ tau_propagator @ pulse_propagator @ tau_propagator) @ x_0)
#         cpmg = (tau_propagator @ pulse_propagator @ tau_propagator @ tau_propagator @ pulse_propagator @ tau_propagator)
#         total_propagator = matrix_power(cpmg, ncyc)
        
#         total_evol = total_propagator @ x_0
#         # print(total_evol[1]/x_0[1])

#         R2eff = -1/self.T_relax * np.log(total_evol[1]/x_0[1])

#         return R2eff

if __name__ == "__main__":
    params ={"R2A": 0,
        "R2B": 0,
        "R1A": 0,
        "R1B": 0,
        "pb": 0,
        "kex": 0,
        "omegaA": 0,
        "omegaB": 0,
        "omega1":2000,
        "omegaRF": 0,
        # assert "M_eqA" in params.keys()
        # assert "M_eqB" in params.keys()
        "B0": 0,}
    cpmg = SimulateCPMGDaiwen(params)
    tot_n = 25
    for i in range(tot_n):
        ncyc = i+1
        print(ncyc, cpmg.calculate_nm(ncyc))
