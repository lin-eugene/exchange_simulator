import numpy as np
import pandas as pd
import pathlib

def filter_out_comments(line):
    if line[0] == '#':
        return False
    else:
        return True
    
def remove_separators(line):
    return line.replace('\n','')

def split_line(line):
    return line.split()

def get_1Hppm(lines):
    for line in lines:
        if 'f01(ppm)' in line:
            H_ppm = line.split()[2]
            H_ppm = float(H_ppm)
            return H_ppm

def get_15Nppm(lines):
    for line in lines:
        if 'f02(ppm)' in line:
            # turn into float
            N_ppm = line.split()[2]
            N_ppm = float(N_ppm)
            return N_ppm

def get_name(path: pathlib.Path):
    return path.name.split('.')[0]


#### get overlap_group
def get_overlap_group(lines: list[str]):
    for idx, line in enumerate(lines):
        if 'Overlap_group' in line:
            info_line = lines[idx+1]
            overlap_group = info_line.split()[2]
            return overlap_group
    return None

def get_input_frequencies(lines: list[str]):
    for idx, line in enumerate(lines):
        if 'Input_frequencies' in line:
            info_line = lines[idx+1]
            info_line = info_line.split()[1:]
            return {'Omega1': float(info_line[0]), 
                    'Omega2': float(info_line[1]), 
                    'Omega3': float(info_line[2])}

    return None

def get_fit_parameters(lines: list[str]):
    for idx, line in enumerate(lines):
        if 'Results of the fit' in line:
            start_line = idx + 3
        
        if '##########' in line:
            end_line = idx - 1
            break
    
    parameter_lines = lines[start_line:end_line+1]
    
    parameter_dict = {}
    for line in parameter_lines:
        line = line.split()
        parameter_name = line[1]
        parameter_value = line[2]
        parameter_error = line[3]

        parameter_dict[f'{parameter_name}_value'] = float(parameter_value)
        parameter_dict[f'{parameter_name}_error'] = float(parameter_error)

    return parameter_dict
            


def filter_large_errors(row):
    error_sum = row['df']['Esd(Int.)'].sum()
    intensity_sum = row['df']['Intensity'].sum()

    if np.abs(error_sum/intensity_sum) > 0.5:
        print(f"Large error found for residue {row['name']}, filtering out.")
        return False
    else:
        return True