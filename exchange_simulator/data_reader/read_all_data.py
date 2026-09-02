from simulators.sim_all import SimulateExchange
from models.models import calculate_exchange_bimolecular
from data_reader.read_HSQC import get_HSQCs, HSQCResidueSeries
from data_reader.read_R2 import get_R2_values, R2ResidueSeries

from data_reader.read_CEST import read_CEST_data, select_residue_CEST_data
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pathlib

class PeakMetadata:
    def __init__(self, 
                 B0,  
                 O3, 
                 N_ppm,
                 conc):
        self.B0 = B0
        self.O3 = O3
        self.N_ppm = N_ppm
        self.O3P = O3/B0
        self.offset_ppm =self.N_ppm-self.O3P
        self.conc=conc

class CESTPeakData:
    def __init__(self, CEST_data, B0, O3, N_ppm, conc):
        self.data = CEST_data
        self.metadata = PeakMetadata(B0, O3, N_ppm, conc)

class CPMGPeakData:
    def __init__(self, CPMG_data, B0, O3, N_ppm, conc):
        self.data = CPMG_data
        self.metadata = PeakMetadata(B0, O3, N_ppm, conc)

class R1rhoPeakData:
    def __init__(self, R1rho_data, B0, O3, N_ppm, conc):
        self.data = R1rho_data
        self.metadata = PeakMetadata(B0, O3, N_ppm, conc)

class R1PeakData:
    def __init__(self, R1_data, B0, O3, N_ppm, conc):
        self.data = R1_data
        self.metadata = PeakMetadata(B0, O3, N_ppm, conc)

class ResidueData:
    def __init__(self, 
                 root_path, 
                 data_df, 
                 construct, 
                 residue,
                 B0):
        self.residue = residue
        CEST_data_df_filtered = data_df[data_df['Protein'] == construct]
        CEST_data_df_filtered = CEST_data_df_filtered[CEST_data_df_filtered['Experiment'] == 'CEST']
        concentration = CEST_data_df_filtered['Actual_conc'].iloc[0]
        CEST_path = root_path / pathlib.Path(CEST_data_df_filtered['Path'].iloc[0])
        CEST_data = read_CEST_data(CEST_path, construct)
        CEST_data_selected = select_residue_CEST_data(CEST_data, residue)

        CEST_N_ppm = next(iter(CEST_data_selected.values()))['N_ppm']
        O3_CEST = 11271.722953353
        self.CEST_data = CESTPeakData(CEST_data_selected, B0, O3_CEST, CEST_N_ppm, concentration)

        CPMG_data_df_filtered = data_df[data_df['Protein'] == construct]
        CPMG_data_df_filtered = CPMG_data_df_filtered[CPMG_data_df_filtered['Experiment'] == 'CPMG']
        CPMG_path = root_path / pathlib.Path(CPMG_data_df_filtered['Path'].iloc[0])
        CPMG_data = pd.read_pickle(CPMG_path / 'out' / 'T2fits.pkl')
        # print(CPMG_path)
        CPMG_data_selected = CPMG_data[CPMG_data['name'] == residue]['raw_df'].iloc[0]
        N_ppm_CPMG = CPMG_data[CPMG_data['name'] == residue]['N_ppm'].iloc[0]
        self.CPMG_data = CPMGPeakData(CPMG_data_selected, B0, O3_CEST, CEST_N_ppm, concentration)

        CW_CPMG_data_df_filtered = data_df[data_df['Protein'] == construct]
        CW_CPMG_data_df_filtered = CW_CPMG_data_df_filtered[CW_CPMG_data_df_filtered['Experiment'] == 'CW-CPMG']
        CW_CPMG_path = root_path / pathlib.Path(CW_CPMG_data_df_filtered['Path'].iloc[0])
        CW_CPMG_data = pd.read_pickle(CW_CPMG_path / 'out' / 'T2fits.pkl')
        CW_CPMG_data_selected = CW_CPMG_data[CW_CPMG_data['name'] == residue]['raw_df'].iloc[0]
        N_ppm_CW_CPMG = CW_CPMG_data[CW_CPMG_data['name'] == residue]['N_ppm'].iloc[0]
        self.CW_CPMG_data = CPMGPeakData(CW_CPMG_data_selected, B0, O3_CEST, CEST_N_ppm, concentration)

        R1rho_data_filtered = data_df[data_df['Protein'] == construct]
        R1rho_data_filtered = R1rho_data_filtered[R1rho_data_filtered['Experiment'] == 'R1rho']
        R1rho_path = root_path / pathlib.Path(R1rho_data_filtered['Path'].iloc[0])
        R1rho_data = pd.read_pickle(R1rho_path / 'out' / 'R1rho_fits.pkl')
        R1rho_data_selected = R1rho_data[R1rho_data['name'] == residue]['fit_df'].iloc[0]
        N_ppm_R1rho = R1rho_data[R1rho_data['name'] == residue]['N_ppm'].iloc[0]
        self.R1rho_data = R1rhoPeakData(R1rho_data_selected, B0, O3_CEST, CEST_N_ppm, concentration)


        R1_data_filtered = data_df[data_df['Protein'] == construct]
        R1_data_filtered  = R1_data_filtered[R1_data_filtered['Experiment'] == 'R1']
        R1_path = root_path / pathlib.Path(R1_data_filtered['Path'].iloc[0])
        R1_data = pd.read_pickle(R1_path / 'out' / 'R1.pkl')
        R1_data_selected = R1_data[R1_data['name'] == residue]['df'].iloc[0]
        self.R1_data = R1PeakData(R1_data_selected, B0, O3_CEST, CEST_N_ppm, concentration)

        HSQC_data_df_filtered = data_df[data_df['Protein'] == construct]
        HSQC_data_df_filtered = HSQC_data_df_filtered[HSQC_data_df_filtered['Experiment'] == 'HSQC']
        HSQCs = HSQCResidueSeries(HSQC_data_df_filtered, residue=residue, path_parent=root_path, )
        self.concs, self.conc_normalised_intensities = HSQCs.get_conc_normalised_intensities()
        self.H_perturbations, self.N_perturbations = HSQCs.get_CSPs()

        R2_data_df_filtered = data_df[data_df['Protein'] == construct]
        R2_data_df_filtered = R2_data_df_filtered[R2_data_df_filtered['Experiment'] == 'CPMG']
        self.R2s = R2ResidueSeries(R2_data_df_filtered, residue=residue, path_parent=root_path, )
        self.delta_R2s, self.delta_R2s_errors = self.R2s.get_delta_R2s()

        self.params_initial ={
                'R2A': 3,
                'R2B': 30,
                'R1A': 1.6,
                'R1B': 1.6,
                'pb': 0.2,
                'kex': 100,
                'omegaA': 0*B0,
                'ppm_diff': 0,
                'B0': 96.2, # 15N
                'I0_R1': 1e6,
                }