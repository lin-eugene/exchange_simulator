
import itertools
from typing import Literal, Optional, Union
import pathlib
import pandas as pd
import logging



def read_metadata(metadata_csv: pathlib.Path,) -> pd.DataFrame:
    metadata_df = pd.read_csv(metadata_csv)
    return metadata_df
   

def find_parameter(parameter:str, acqus_filepath: pathlib.Path) -> Union[float, None]:
            if not acqus_filepath.exists():
                return None

            with open(acqus_filepath, 'r') as file:
                for line in file:
                    if parameter in line:
                        parameter_value = line.split(' ')[1]
                        parameter_value = float(parameter_value)
                        return parameter_value
                    
            return None


def find_peak_list(exp_directory: Union[pathlib.Path, str],) -> pd.DataFrame:
    exp_directory_root = pathlib.Path(exp_directory)

    if exp_directory_root.name == 'raw':
        exp_directory_root = exp_directory_root.parent
    if exp_directory_root.name == 'out':
        exp_directory_root = exp_directory_root.parent
        

    peak_list_file = exp_directory_root / 'out' / 'correlate.3'

    peak_list = pd.read_csv(peak_list_file, sep='\t', header=None,)
    
    rename = {
                0: 'residue',
                1: 'H_ppm',
                2: 'N_ppm',
                3: 'intensity',
              }
    
    peak_list = peak_list.rename(columns=rename)
    return peak_list

def get_peak_list(exp_directory: Union[pathlib.Path, str],) -> pd.DataFrame:
    peak_list = find_peak_list(exp_directory)
    acqus_filepath = pathlib.Path(exp_directory) / 'raw' / 'acqus'
    RG = find_parameter('##$RG', acqus_filepath)
    NS = find_parameter('##$NS', acqus_filepath)
    O1 = find_parameter('##$O1', acqus_filepath)
    O3 = find_parameter('##$O3', acqus_filepath)
    B0_H = find_parameter('##$SFO1', acqus_filepath)
    B0_N = find_parameter('##$SFO3', acqus_filepath)
    peak_list['normalised_intensity'] = peak_list['intensity'] / NS
    return peak_list, RG, NS, O1, O3, B0_H, B0_N

class Spectrum:
    def __init__(self, 
                 exp_directory,
                 concentration):
        self.peak_list, self.RG, self.NS, self.O1, self.O3, self.B0_H, self.B0_N = get_peak_list(exp_directory)
        self.concentration = concentration
        self.peak_list['conc_normalised_intensity'] = self.peak_list['normalised_intensity'] / self.concentration

    def get_residue(self, residue: str):
        residue_data = self.peak_list[self.peak_list['residue'] == residue]
        return residue_data

def get_HSQCs(metadata_df, path_parent: Optional[pathlib.Path]=None) -> list[Spectrum]:
    HSQCs = []

    for index, row in metadata_df.iterrows():
        exp_directory = pathlib.Path(row['Path'])
        if path_parent is not None:
            exp_directory = path_parent / exp_directory
        concentration = row['Actual_conc']
        spectrum = Spectrum(exp_directory, concentration)
        HSQCs.append(spectrum)

    return HSQCs
         
         
class HSQCResidueSeries:
    def __init__(self,
                metadata_df: pd.DataFrame,
                residue: str,
                path_parent: Optional[pathlib.Path]=None,
                ):
            self.HSQCs = get_HSQCs(metadata_df, path_parent)
            # print(self.HSQCs)
            self.residue = residue

            self.get_residue_series()

    def get_residue_series(self):
        self.residue_series = []
        for hsqc in self.HSQCs:
            residue_data = hsqc.get_residue(self.residue)
            self.residue_series.append({'spectrum': hsqc,
                                   'residue_data': residue_data})

    def get_conc_normalised_intensities(self):
        self.conc_normalised_intensities = []
        self.concs = []
        for item in self.residue_series:
            residue_data = item['residue_data']
            if not residue_data.empty:
                conc_normalised_intensity = residue_data['conc_normalised_intensity'].values[0]
                conc = item['spectrum'].concentration
                self.conc_normalised_intensities.append(conc_normalised_intensity)
                self.concs.append(conc)
        return self.concs, self.conc_normalised_intensities

    def get_CSPs(self):
        reference_conc = min(self.concs)
        # print(reference_conc)
        self.H_perturbations = {}
        self.N_perturbations = {}

        for item in self.residue_series:
            residue_data = item['residue_data']
            if not residue_data.empty:
                conc = item['spectrum'].concentration
                if conc == reference_conc:
                    ref_H_ppm = residue_data['H_ppm'].values[0]
                    ref_N_ppm = residue_data['N_ppm'].values[0]
                    # print(ref_H_ppm, ref_N_ppm)

        for item in self.residue_series:
            residue_data = item['residue_data']
            conc = item['spectrum'].concentration
            H_ppm = residue_data['H_ppm'].values[0]
            N_ppm = residue_data['N_ppm'].values[0]
            H_perturbation = H_ppm - ref_H_ppm
            N_perturbation = N_ppm - ref_N_ppm
            self.H_perturbations[conc] = H_perturbation
            self.N_perturbations[conc] = N_perturbation
        return self.H_perturbations, self.N_perturbations
            
         



# def add_metadata_to_peak_list(exp_directory: pathlib.Path,
#                               peak_list: pd.DataFrame,
#                               metadata_df: pd.DataFrame) -> pd.DataFrame:
    


#     RGs = []
#     NSs = []

#     for index, row in metadata_df.iterrows():
#         acqus_filepath = pathlib.Path(row['directory']) / 'raw' / 'acqus'
#         RG = find_parameter('##$RG', acqus_filepath)
#         NS = find_parameter('##$NS', acqus_filepath)
#         RGs.append(RG)
#         NSs.append(NS)

#     metadata_df['receiver_gain'] = RGs
#     metadata_df['ns'] = NSs
#     logging.debug(peak_list)
#     logging.debug(exp_directory)
#     logging.debug(metadata_df)
#     logging.debug(metadata_df.directory.iloc[0] == exp_directory)
#     logging.debug(metadata_df.directory.iloc[0], exp_directory)
#     peak_list['directory'] = str(exp_directory)
#     try:
#         peak_list['intended_conc'] = metadata_df[metadata_df['directory'] == str(exp_directory)]['intended_conc'].values[0]
#     except KeyError:
#         pass
#     peak_list['construct'] = metadata_df[metadata_df['directory'] == str(exp_directory)]['construct'].values[0]
#     peak_list['ddx4_conc'] = metadata_df[metadata_df['directory'] == str(exp_directory)]['ddx4_conc'].values[0]
#     peak_list['nacl_conc'] = metadata_df[metadata_df['directory'] == str(exp_directory)]['nacl_conc'].values[0]
#     peak_list['temp'] = metadata_df[metadata_df['directory'] == str(exp_directory)]['temp'].values[0]
#     peak_list['pH'] = metadata_df[metadata_df['directory'] == str(exp_directory)]['pH'].values[0]
#     peak_list['receiver_gain'] = metadata_df[metadata_df['directory'] == str(exp_directory)]['receiver_gain'].values[0]
#     peak_list['ns'] = metadata_df[metadata_df['directory'] == str(exp_directory)]['ns'].values[0]
#     peak_list['exp'] = metadata_df[metadata_df['directory'] == str(exp_directory)]['exp'].values[0]
#     # peak_list['ppm_shift_methyl'] = metadata_df[metadata_df['exp_no'] == exp_no]['ppm_shift_methyl'].values[0]

#     return peak_list