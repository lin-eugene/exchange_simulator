import numpy as np
import odeintw

def fid(x, t, A):

    return A @ x

def simulate_spectrum(liouvillian_matrix,
                        params):
        """
        Simulate the evolution of the magnetization using matrix exponentiation.
    
        Parameters:
        liouvillian_matrix (np.ndarray): The Liouvillian matrix representing the system.
        c_init (np.ndarray): The initial state vector.
        times (list or np.ndarray): The time points at which to evaluate the state.
    
        Returns:
        np.ndarray: The state vectors at the specified time points.
        """
        B0 = params['B0']
        t_array = np.linspace(0, 0.5, 5000)
        c_init = np.array([1-params['pb'], params['pb']], dtype=np.complex128)

        solver = odeintw.odeintw(fid, c_init, t_array, args = (liouvillian_matrix,))
        #sum the two magnetizations to get total signal
        M_total = solver[:,0] + solver[:,1]
        fft_M = np.fft.fftshift(np.fft.fft(M_total))
        freqs = np.fft.fftshift(np.fft.fftfreq(len(t_array), d=t_array[1]-t_array[0]))
        freqs_ppm = freqs / B0
        fft_M /= len(fft_M)


        return {'freqs_ppm': freqs_ppm,
                'spectrum': fft_M}