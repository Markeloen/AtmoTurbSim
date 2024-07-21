import numpy
import matplotlib.pylab as plt
from tqdm import tqdm, trange, tqdm_notebook
from astropy.io import fits
import time
import imageio
import os
import scipy
# %matplotlib inline


from phasescreen_github import *
from infinitephasescreen import *

from matplotlib import animation, rc




def main():
    # Set up parameters for creating phase screens
    
    nx_size = 32
    D = 10
    pxl_scale = D/nx_size
    r0 = 2
    L0 = 10
    # wind_speed = 10 #m/s 
    # n_tests = 25 # 16
    # n_scrns = 100
    stencil_length_factor = 32

    s = time.time()
    phase_screen = PhaseScreenKolmogorov(nx_size, pxl_scale, r0, L0, stencil_length_factor=stencil_length_factor)
    # print(round(time.time()-s,2))
    # plt.figure()
    # plt.imshow(phase_screen.scrn)
    # cbar = plt.colorbar()
    # cbar.set_label('Wavefront deviation (radians)', labelpad=8)


    #save it
    filenames = []
    frames = 10

    output_mat = []


    for i in tqdm(range(frames)):
        # print(f"Calculating frame: {i+1} ........")
        s = time.time()
        phase_screen.add_row()
        # print(round(time.time()-s,2))
        plt.imshow(numpy.transpose(phase_screen.scrn))
        plt.draw()
        plt.pause(0.5)
# 

        filename = f'frame_{i}.png'
        plt.savefig(filename)
        # plt.close()
        filenames.append(filename)
        output_mat.append(numpy.transpose(phase_screen.scrn))
    # output_mat = numpy.array(output_mat)
    # print(f"The size of output is: {output_mat.shape}")
    # Create a GIF
    with imageio.get_writer('my_animation.gif', mode='I', duration=0.1) as writer:
        for filename in filenames:
            image = imageio.v2.imread(filename)
            writer.append_data(image)

    # Remove f iles
    for filename in filenames:
        os.remove(filename)
    # output_mat = numpy.transpose(output_mat,(1, 2, 0))
    # scipy.io.savemat(f'phase_screens_matrix_{nx_size}_{frames}.mat', {f'phase_screens_matrix_{nx_size}_{frames}':output_mat})

if __name__ == "__main__":
    main()