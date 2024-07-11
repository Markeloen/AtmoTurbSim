import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm, trange, tqdm_notebook

from propagation.basic_funcs import *
from phase_screens.phasescreen_github import *
from Scripts.Layer import Layer

def str_fnc2_ft(ph, mask, delta):
    N = ph.shape[0]
    ph = ph * mask

    
    P = ft2(ph, delta)
    S = ft2(ph**2, delta)
    W = ft2(mask, delta)

    delta_f = 1 / (N * delta)
    w2 = ift2(W * np.conj(W), delta_f)

    # np.real and np.conj are used for real part and complex conjugate respectively
    D = 2 * ift2(np.real(S * np.conj(W)) - np.abs(P)**2, delta_f) / w2 * mask
    
    
    
    return np.abs(D)

def str_fnc2_ft2(ph, mask, delta):

    N = (ph.shape)[0]
    ph = ph * mask

    P = ft2(ph, delta)
    S = ft2(ph**2, delta)
    W = ft2(mask, delta)

    delta_f = 1/(N*delta)
    w2 = ift2(W*tf.math.conj(W), delta_f)

    D = 2 * ift2(tf.cast(tf.math.real(S*tf.math.conj(W)) - tf.abs(P)**2, tf.complex64), delta_f) / w2 * mask
    return tf.abs(D)

def structure_function(phase_screen, delta):
    """ Calculate the structure function from a phase screen """
    N = phase_screen.shape[0]
    D = np.zeros((N//2,))
    for i in range(N//2):
        shift = np.roll(phase_screen, i, axis=0)
        D[i] = np.mean((phase_screen - shift)**2)
    r = np.arange(N//2) * delta
    return r, D


def coh_val_monte_carlo(r0, N, delta, nreal, l0 = .01, L0 = 100, number_of_extrusions = 100):
    avg_MDOC2 = 0
    for _ in tqdm(range(nreal)):
        layer = Layer(N, delta, r0, L0)
        for _ in tqdm(range(number_of_extrusions)):
            layer.extude_return_scrn()
        avg_MDOC2 += layer.return_MDOC2() / nreal
    theoretical = layer.retrun_modulus_function_xaxis()[1]
    x = layer.retrun_modulus_function_xaxis()[2]
    
    plt.plot(x, avg_MDOC2, label='Simulation')
    plt.plot(x, theoretical, label='Theoretical')
    plt.title(f'Monte Carlo Validation (r0={r0}, N={N}, delta={delta})', fontsize=10)
    plt.xlabel('r (m)')
    plt.ylabel('MDOC2 (r)')
    plt.legend()
    plt.show()
        
    
    

if __name__ == "__main__":
    # # !!! Debug this section !!!
    
    N = 256
    delta = 0.01
    L0 = 100
    l0 = 0.001
    r0 = 1
    # L = 16
    # delta = L / N
    # w = 2
    
    x = np.arange(-N/2., N/2.)
    # print(len(x))
    [nx, ny] = np.meshgrid(x, x)
    
    
    xn = nx * delta
    yn = ny * delta
    
    # A = rect(xn, 2) * rect(yn, 2)
    
    # mask = circ(xn, yn , N * delta)
    mask = np.ones((N, N))
    avg_str_fnc = 0
    N_realization = 10
    
    for i in range(N_realization):
        ph = ft_sh_phase_screen(r0, N, L0, l0, delta, seed=i)
        avg_str_fnc += str_fnc2_ft(ph, mask, delta) / N_realization
        # avg_str_fnc += str_fnc2_ft(A, mask, 1) / N_realization
    
    # plt.plot( np.linspace(0, 128, 128)*delta,np.abs(avg_str_fnc[N//2, N//2:]))    
    
    x = np.linspace(0, 12 * r0, N//2)
    theo_struc = 6.88 * (np.abs(x)/r0) ** (5/3)
    plt.plot(x,  avg_str_fnc[N//2, N//2:], label='Simulation')
    plt.show()
    # comment args guide here here:
    # plot_list = [[128, 0.2, 0.02], [128, 0.5, 0.05], [256, 1, 0.2]]

    # for params in plot_list:
    #     N, r0, delta = params
    #     coh_val_monte_carlo(r0, N, delta, 100, 0.01, 25, 5 * N)
    #     plt.savefig(f'plot_{N}_{r0}_{delta}.png')
    #     plt.close()