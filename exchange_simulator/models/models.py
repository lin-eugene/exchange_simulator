import numpy as np

def calculate_exchange_bimolecular(A_tot,
                              Kd,
                              koff):
    """
    A + A <—> C
    """

    A = Kd * (1+(A_tot/Kd))**0.5 -Kd
    C = A**2/Kd
    pb = C / A_tot

    kex = koff*A/Kd + koff
    # kon = koff / Kd

    return pb, kex