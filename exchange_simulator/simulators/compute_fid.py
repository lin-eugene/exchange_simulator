import numpy as np
from scipy.linalg import expm
from collections import defaultdict

def calculate_fid(free_precession_liouvillian,
                  c_init,
                  dwell_time,
                  duration,):
    
    n_points = int(duration / dwell_time)

    mat_exp = expm(free_precession_liouvillian * dwell_time)

    c_current = c_init.copy()
    fid = defaultdict(list)
    for i in range(n_points):
        fid['time'].append(i * dwell_time)
        #FIX THIS
        c_current = mat_exp @ c_current
        fid['I'].append(c_current)

    fid['I'] = np.array(fid['I'])
    
    return fid

def compute_spectrum(fid):
    n_points = len(fid['time'])
    signal = fid['I'][:,0] + 1j*fid['I'][:,1] + fid['I'][:,3] + 1j*fid['I'][:,4]
    # print(fid['I'].shape)
    # print(signal.shape)
    intensities = np.fft.fftshift(np.fft.fft(signal))
    frequencies = np.fft.fftshift(np.fft.fftfreq(n_points, d=(fid['time'][1]-fid['time'][0])))
    spectrum = {'frequencies': frequencies, 
            'intensity': intensities}
    return spectrum

def calculate_max_intensity(spectrum):
    # I = np.max(np.abs(spectrum['intensity']))
    I = np.sum(np.abs(spectrum['intensity']))
    return I



    