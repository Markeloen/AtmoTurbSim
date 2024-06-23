import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from propagation.basic_funcs import *
from phase_screens.phasescreen_github import *

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
    
    return D

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


N = 256
delta = 0.001
L0 = 100
l0 = 0.01
r0 = 0.01

x = np.arange(-N/2., N/2.)
[nx, ny] = np.meshgrid(x, x)

xn = nx * delta
yn = ny * delta

mask = circ(xn, yn , N * delta)

avg_str_fnc = 0
N_realization = 100

for i in range(N_realization):
    ph = ft_sh_phase_screen(r0, N, L0, l0, delta, seed=i)
    avg_str_fnc += str_fnc2_ft(ph, mask, delta) / N_realization
    
    
x = np.linspace(0, 12 * r0, N//2)
theo_struc = 6.88 * (np.abs(x)/r0) ** (5/3)
plt.plot(x,  avg_str_fnc[N//2, N//2:], label='Simulation')
plt.show()