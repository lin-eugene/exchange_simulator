import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib
import seaborn as sns
from itertools import repeat
import re
from collections import defaultdict

B1_dict = {'WT': {'CEST-0':379.08, 'CEST-4':544.95, 'CEST-7': 801.72},
           'CS': {'CEST-0':379.69, 'CEST-3':545.00, 'CEST-7': 802.01},
           '14FtoA': {'CEST-0':379.36, 'CEST-3':540.77, 'CEST-7': 796.15}}

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



def read_path(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    H_ppm = get_1Hppm(lines)
    N_ppm = get_15Nppm(lines)
    lines = list(filter(filter_out_comments, lines))
    lines = list(map(remove_separators, lines))
    lines = np.array(list(map(split_line, lines)),dtype=float)
    # print(lines)
    df = pd.DataFrame(lines)

    df.columns = ['N_offsetHz', 'Intensity', 'Esd(Int.)']

    ref_offset = df['N_offsetHz'].min()
    I0 = df.loc[df['N_offsetHz'] == ref_offset, 'Intensity'].values[0]
    df['I/I0'] = df['Intensity']/I0
    df = df.drop(df[df['N_offsetHz'] == ref_offset].index)
    df = df.reset_index(drop=True)

    #find duplicate offsets and average them
    duplicate = df.duplicated(subset=['N_offsetHz'])
    duplicate_offset = df['N_offsetHz'][duplicate].unique()[0]
    duplicate_rows = df[df['N_offsetHz'] == duplicate_offset]
    duplicate_mean = duplicate_rows.mean()
    duplicate_std = duplicate_rows.std()
    df['I/I0_error'] = duplicate_std['I/I0']
    df = df.drop(df[df['N_offsetHz'] == duplicate_offset].index)

    duplicate_df = pd.DataFrame(columns=df.columns)
    duplicate_df.loc[0] = [duplicate_mean['N_offsetHz'], duplicate_mean['Intensity'], duplicate_mean['Esd(Int.)'], duplicate_mean['I/I0'], duplicate_std['I/I0']]
    df = pd.concat([df, duplicate_df], ignore_index=True)
    df = df.sort_values(by='N_offsetHz')

    name = path.name.split('.')[0]
    residue_number = int(re.findall(r'\d+', name)[0])
    residue_name = re.findall(r'[A-Za-z]+', name)[0]
    residueID = f"{residue_name}{residue_number}"

    #average I/I0 at -20000 and 20000
    if -20000 and 20000 in df['N_offsetHz'].values:
        I0_values = df.loc[df['N_offsetHz'].isin([-20000, 20000]), 'I/I0'].to_numpy()
        R1s = -np.log(I0_values)/0.7
        # print(f'{R1s=}')


    return {'name': path.name.split('.')[0], 'df': df, 'I0': I0, 'H_ppm': H_ppm, 'N_ppm': N_ppm, 'residueID': residueID, 'residue_no': residue_number, 'R1_list': R1s}

def read_CEST_data(path, construct):
    data_path = path / 'out' / 'data_edited'
    CEST_directories = list(data_path.glob('**/CEST*'))
    # print(CEST_directories)
    CEST_directories = [d for d in CEST_directories if d.is_dir()]
    CEST_directories = sorted(CEST_directories, key=lambda x: int(re.findall(r'\d+', x.name)[0]), reverse=True)
    
    CEST_outs = {}
    for d in CEST_directories:
        out_files = list(d.glob('**/*.out'))
        out_files = sorted(out_files, key=lambda x: int(re.findall(r'\d+', x.name)[0]))
        CEST_outs[d.name] = out_files

    CEST_dict = {}
    for i, (key, out_files) in enumerate(CEST_outs.items()):
        # print(f"Processing {key} with {len(out_files)} files")
        data = list(map(read_path, out_files))
        CEST_dict[B1_dict[construct][key]] = data

    return CEST_dict

def select_residue_CEST_data(CEST_data, residue):
    residue_data = {}
    for B1, data_list in CEST_data.items():
        for data in data_list:
            if data['name'] == residue:
                residue_data[B1] = data
                break
    return residue_data
