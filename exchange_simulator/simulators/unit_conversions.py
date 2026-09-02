import numpy as np

def convert_ppm_to_rads(ppm,
                        B0MHz,
                        gamma):
        return ppm * B0MHz * gamma * 2 * np.pi

def convert_Hz_to_rads(Hz):
        return Hz * 2 * np.pi

def convert_Hz_to_ppm(Hz,
                      B0MHz,
                      gamma):
        return Hz / (B0MHz * gamma * 2 * np.pi)