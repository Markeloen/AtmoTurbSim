import numpy
import imageio
from tqdm import tqdm
from phase_screens.infinitephasescreen  import PhaseScreenKolmogorov
import os   
import matplotlib.pyplot as plt
from propagation.basic_funcs import *


def create_gif_from_phase_screens(nx_size, pxl_size, r0, L0, stencil_length_factor, frames):
    phase_screen = PhaseScreenKolmogorov(nx_size, pxl_size, r0, L0, stencil_length_factor=stencil_length_factor)
    filenames = []
    frames_directory = os.path.join(os.path.dirname(__file__), 'imageFolder')
    for i in tqdm(range(frames)):
        phase_screen.add_row()
        plt.clf()
        plt.imshow(phase_screen.scrn)
        plt.colorbar()
        plt.draw()
        filename = os.path.join(frames_directory, f'frame_{i}.png')  # Construct full path for saving frame
        plt.savefig(filename)
        filenames.append(filename)

    gif_filename = os.path.join(os.path.dirname(__file__), 'imageFolder', f'anime{nx_size}by{nx_size}_pxl-scale={pxl_size}_r0={r0}.gif')

    with imageio.get_writer(gif_filename, mode='I', duration=0.001) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

    # remove png files  
    for filename in set(filenames):
        os.remove(filename)


def str_fcn2_ft(ph, mask, delta):
    N = ph.shape[0]
    ph = ph * mask

    P = ft2(ph, delta)
    S = ft2(np.square(ph), delta)
    W = ft2(mask, delta)

    delta_f = 1/(N*delta)


    w2 = ift2(W*np.conj(W), delta_f)

    D = 2 * ift2(np.real(S*np.conj(W)) - np.square(np.abs(P)), delta_f) / w2 * mask
    return D

    