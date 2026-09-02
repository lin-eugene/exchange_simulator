import pathlib
from typing import Optional
import pandas as pd

def fetch_R2_data(exp_directory):
    exp_directory = pathlib.Path(exp_directory)
    R2_file = exp_directory / 'out' / 'T2fits.csv'
    R2_data = pd.read_csv(R2_file)
    return R2_data


class Spectrum:
    def __init__(self, 
                 exp_directory,
                 concentration):
        self.R2_data = fetch_R2_data(exp_directory)
        self.concentration = concentration
        # self.R2_data['conc_normalised_intensity'] = self.R2_data['normalised_intensity'] / self.concentration

    def get_residue(self, residue: str):
        residue_data = self.R2_data[self.R2_data['name'] == residue]
        return residue_data
    

def get_R2_values(metadata_df: pd.DataFrame, 
                  path_parent: Optional[pathlib.Path] = None) -> list[Spectrum]:
    R2_values = []
    for index, row in metadata_df.iterrows():
        exp_directory = pathlib.Path(row['Path'])
        if path_parent is not None:
            exp_directory = path_parent / exp_directory
        concentration = row['Actual_conc']
        spectrum = Spectrum(exp_directory, concentration)
        R2_values.append(spectrum)
    return R2_values


class R2ResidueSeries:
    def __init__(self,
                 metadata_df: pd.DataFrame,
                 residue: str,
                 path_parent: Optional[pathlib.Path]=None,
                 ):
        self.R2_values = get_R2_values(metadata_df, path_parent)
        self.residue = residue

        self.get_residue_series()

    def get_residue_series(self):
        self.residue_series = []
        for r2_value in self.R2_values:
            residue_data = r2_value.get_residue(self.residue)
            # print(residue_data)
            self.residue_series.append({'spectrum': r2_value,
                                   'residue_data': residue_data})
    
    def get_concs(self):
        self.concs = []
        for item in self.residue_series:
            residue_data = item['residue_data']
            # print(residue_data)
            if not residue_data.empty:
                conc = item['spectrum'].concentration
                self.concs.append(conc)

        # print(self.concs)
        return self.concs
    
    def get_delta_R2s(self):
        reference_concs = min(self.get_concs())
        self.delta_R2s = {}
        self.delta_R2s_errors = {}
        for item in self.residue_series:
            residue_data = item['residue_data']
            if not residue_data.empty:
                conc = item['spectrum'].concentration
                if conc == reference_concs:
                    R2_ref = residue_data['R2'].values[0]
                    R2_ref_error = residue_data['R2_error'].values[0]
        
        for item in self.residue_series:
            residue_data = item['residue_data']
            conc = item['spectrum'].concentration
            if not residue_data.empty:
                R2 = residue_data['R2'].values[0]
                delta_R2 = R2 - R2_ref
                delta_R2_error = residue_data['R2_error'].values[0] + R2_ref_error
                self.delta_R2s[conc] = delta_R2
                self.delta_R2s_errors[conc] = delta_R2_error

            
        # get reference R2

        return self.delta_R2s, self.delta_R2s_errors