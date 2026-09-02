import lmfit
import numpy as np
from simulators.sim_all import SimulateExchange
from data_reader.read_all_data import ResidueData

# Datasets

def gen_fit_params(residue_dataset: list[ResidueData]):
    params = lmfit.Parameters()

    for dataset in residue_dataset:
        residue_name = dataset.residue
        params.add(f'R2A_{residue_name}', value = dataset.params_initial['R2A'], vary=True, min=0)
        params.add(f'R2B_{residue_name}', value = dataset.params_initial['R2B'], min=0)
        params.add(f'R1A_{residue_name}', value = dataset.params_initial['R1A'], min=0)
        params.add(f'R1B_{residue_name}', value = dataset.params_initial['R1B'],
                   expr=f'R1A_{residue_name}'
                   )
        # params.add(f'R2fudge_{residue_name}', value=dataset.params_initial['R2fudge'])
        params.add(f'omegaA_{residue_name}', value = dataset.params_initial['omegaA'], vary=False)
        params.add(f'ppm_diff_{residue_name}', value = dataset.params_initial['ppm_diff'], vary=True)
        params.add(f'pb_{residue_name}', value = dataset.params_initial['pb'],min=0)
        params.add(f'kex_{residue_name}', value = dataset.params_initial['kex'],min=0)
        params.add(f'I0_R1_{residue_name}', value = dataset.params_initial['I0_R1'],min=0)
        
        params.add(f'B0_{residue_name}', value = dataset.params_initial['B0'], vary=False)

    #setting global parameters
    for idx, dataset in enumerate(residue_dataset):
        if idx == 0:
            continue
        residue_name = dataset.residue
        params[f'pb_{residue_name}'].expr = f'pb_{residue_dataset[0].residue}'
        params[f'kex_{residue_name}'].expr = f'kex_{residue_dataset[0].residue}'
    return params


class SimulateDataset:
    def __init__(self,
                 params,
                 dataset):
        
        residue_name = dataset.residue
        
        self.params = {
                'R2A': params[f'R2A_{residue_name}'],
                'R2B': params[f'R2B_{residue_name}'],
                'R1A': params[f'R1A_{residue_name}'],
                'R1B': params[f'R1B_{residue_name}'],
                'omegaA': params[f'omegaA_{residue_name}'],
                'omegaB': params[f'omegaA_{residue_name}'] + params[f'ppm_diff_{residue_name}']*params[f'B0_{residue_name}'],
                'pb': params[f'pb_{residue_name}'],
                'kex': params[f'kex_{residue_name}'],
                'B0': params[f'B0_{residue_name}'], # 15N
                'I0_R1': params[f'I0_R1_{residue_name}'],
                # 'R2fudge': params[f'R2fudge_{residue_name}']
                 }
        
        self.sim = SimulateExchange(self.params)

    def simulate_cest_dataset(self, 
                            omega_RFs_CEST: list, 
                            omega1_CEST: float, 
                            time_CEST: float):
        
        self.I_I0 = self.sim.simulate_CEST(omega_RFs_CEST,
                                                   omega1_CEST,
                                                   time_CEST)
    
    def simulate_r1rho_dataset(self,
                               omega_RFs_R1rho: list,
                            T_relax: float,
                            omega1_R1rho: float):
        
        self.r1rho_profile = self.sim.simulate_R1rho(omega_RFs_R1rho,
                                                     T_relax=T_relax,
                                                     omega1=omega1_R1rho)
        self.r1rho_profile_inhomogeneous_B1 = self.sim.simulate_R1rho_inhomogeneous_B1(omega_RFs_R1rho,
                                                     T_relax=T_relax,
                                                     omega1=omega1_R1rho)
        
    
    def simulate_cpmg_dataset(self,
                             ncycs: list,
                             T_relax: float):
        ncycs = np.array(ncycs, dtype=int)
        self.nu_cpmgs, cpmg_profile = self.sim.simulate_CPMG_numerical(ncycs,
                                                                            omega1=1/(4*39e-6),
                                                                        T_relax=T_relax)
        
        self.cpmg_profile = np.array(cpmg_profile) #+ self.params['R2fudge']
    
    def simulate_r1_dataset(self,
                            taus: list):
        
        self.r1_profile = self.sim.simulate_R1(taus)
        
        

def objective(params: lmfit.Parameters, 
              datasets: list[ResidueData]):
    residuals = []
    for dataset in datasets:
        sim_dataset = SimulateDataset(
            params = params,
            dataset = dataset
        )

        # Simulate CEST
        for omega1_CEST, data, in dataset.CEST_data.data.items():
            omega_RFs_CEST = data['df']['N_offsetHz'].to_numpy()
            time_CEST = 0.7  # s
            sim_dataset.simulate_cest_dataset(
                omega_RFs_CEST=omega_RFs_CEST,
                omega1_CEST=omega1_CEST,
                time_CEST=time_CEST
            )
            i_i0_sim = sim_dataset.I_I0
            i_i0_exp = data['df']['I/I0'].to_numpy()
            i_i0_error = data['df']['I/I0_error'].to_numpy()
            resid_cest = (i_i0_exp - i_i0_sim) / i_i0_error
            residuals.append(resid_cest)
        # print(residuals)

        # Simulate R1rho
        omega_RFs_R1rho = dataset.R1rho_data.data['N_offsetHz'].to_numpy()
        T_relax = 0.16  # s
        omega1_R1rho = 2000  # Hz
        sim_dataset.simulate_r1rho_dataset(
            omega_RFs_R1rho=omega_RFs_R1rho,
            T_relax=T_relax,
            omega1_R1rho=omega1_R1rho
        )
        r1rho_sim = sim_dataset.r1rho_profile.R1rhos.values()
        r1rho_exp = dataset.R1rho_data.data['R_eff_fit'].to_numpy()
        r1rho_error = dataset.R1rho_data.data['R_eff_fit_error'].to_numpy()
        resid_r1rho = (r1rho_exp - np.array(list(r1rho_sim))) / r1rho_error
        residuals.append(resid_r1rho)
        
        reff_sim = sim_dataset.r1rho_profile.Reffs
        reff_exp = dataset.R1rho_data.data['R_eff/sin2_theta'].to_numpy()
        reff_error = dataset.R1rho_data.data['R_eff/sin2_theta_error'].to_numpy()
        resid_reff = (reff_exp - np.array(list(reff_sim))) / reff_error
        residuals.append(resid_reff)


        cfast_sim = sim_dataset.r1rho_profile.cfasts.values()
        cfast_exp = dataset.R1rho_data.data['c_fast'].to_numpy()
        cfast_error = dataset.R1rho_data.data['c_fast_error'].to_numpy()
        resid_cfast = (cfast_exp - np.array(list(cfast_sim))) / cfast_error
        residuals.append(resid_cfast)


        # print(residuals)

        # Simulate CPMG
        nu_cpmgs = dataset.CW_CPMG_data.data['nu_cpmg'].to_numpy()
        ncycs = dataset.CW_CPMG_data.data['ncyc'].to_numpy()
 
        T_relax_CPMG = 0.08
        sim_dataset.simulate_cpmg_dataset(
            ncycs=ncycs,
            T_relax=T_relax_CPMG
            )
        R2eff_exp = dataset.CW_CPMG_data.data['R2eff'].to_numpy()
        R2eff_error = dataset.CW_CPMG_data.data['R2eff_error'].to_numpy()
        resid_cpmg = (R2eff_exp - sim_dataset.cpmg_profile) #/ R2eff_error
        residuals.append(resid_cpmg)

        # Simulate R1
        # taus_R1 = dataset.R1_data.data['tau'].to_numpy()
        # sim_dataset.simulate_r1_dataset(
        #     taus=taus_R1
        # )
        # I0 = sim_dataset.params['I0_R1']
        # R1_exp = dataset.R1_data.data['Intensity'].to_numpy() / I0
        # R1_error = (dataset.R1_data.data['Intensity_error'].to_numpy()/dataset.R1_data.data['Intensity'].to_numpy())  * I0
        # resid_r1 = (R1_exp - sim_dataset.r1_profile)  / R1_error
        # residuals.append(resid_r1)
     
        conc = dataset.CW_CPMG_data.metadata.conc
        # delta_R2_exp = dataset.delta_R2s[conc]
        # delta_R2_error = dataset.delta_R2s_errors[conc]
        # delta_R2_sim = sim_dataset.sim.deltaR2
        # resid_delta_R2 = 10*(delta_R2_exp - delta_R2_sim) # / delta_R2_error
        # residuals.append([resid_delta_R2])

        # CSP = dataset.N_perturbations[conc]
        # sim_dataset.sim.simulate_intensities()
        # CSP_sim = sim_dataset.sim.nfreq
        # resid_CSP = (CSP-CSP_sim)
        # residuals.append([resid_CSP])





    # print(np.concatenate(residuals).sum())
    return np.concatenate(residuals)

def calculate_fits(params: lmfit.Parameters,
                   datasets: list[ResidueData]):
    result = lmfit.minimize(objective, 
                            params, 
                            args=(datasets,), 
                            method='least_squares',
                            max_nfev=1000)
    return result


# def cest_dataset(params,
#                  residue, 
#                  offsets):
#     pass

# class R1rhoDataset:
#     def __init__(self,
#                  params,
#                   residue, 
#                   offsets):
#         pass

# def cpmg_dataset(params,
#                  residue, 
#                  nu_cpmgs):
#     pass

# def r1_dataset(params,
#                residue, 
#                taus):
#     pass



####
"""
def gaussian(x, A, mu, sigma, y_offset):
    return A*np.exp(-(x-mu)**2/(2*sigma**2))+y_offset

def gaussian_dataset(params, i, x):
    A = params[f'A_{i}']
    mu = params[f'mu_{i}']
    sigma = params[f'sigma_{i}']
    y_offset = params[f'y_offset_{i}']
    return gaussian(x, A, mu, sigma, y_offset)

def objective(params, data):
    residuals = []
    for i, datum in enumerate(data):
        dat = datum['fit_df']
        x = np.array(dat['N_offsetHz'])
        y = np.array(dat['R_eff_fit'])
        resid = (y - gaussian_dataset(params, i, x))/np.array(dat['R_eff_fit_error'])
        residuals.append(resid)
    return np.concatenate(residuals)

def gen_fit_params(metadata):
    params = lmfit.Parameters()
    for i, row in metadata.iterrows():
        params.add(f'A_{i}', value=(row['max-min_R_eff'])/2, min=0)
        # params.add(f'mu_{i}', value=row['N_resonance_offset_Hz'], min=row['N_resonance_offset_Hz']-2000, max=row['N_resonance_offset_Hz']+2000)
        #constrain mu to be within the range of N_offsetHz
        params.add(f'mu_{i}', value=row['N_resonance_offset_Hz'], vary=False)
        params.add(f'sigma_{i}', value=1000, min=100, max=5000)
        params.add(f'y_offset_{i}', value=row['fit_df']['R_eff_fit'].min(), min=0)
    
    for idx in range(len(metadata)-1):
        params[f'sigma_{idx+1}'].expr = 'sigma_0'

    return params

gaussian_fit_params = gen_fit_params(fit_df)
result = lmfit.minimize(objective, gaussian_fit_params, args=(fit_data,))
lmfit.report_fit(result)
"""