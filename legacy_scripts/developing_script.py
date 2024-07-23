from phase_screens.infinitephasescreen import *
from phase_screens.phasescreen_github import *
from phase_screens.turb import *
from propagation.Propagator import *
from propagation.basic_funcs import *
from PIL import Image
import tensorflow
from tqdm import tqdm, trange, tqdm_notebook
import matplotlib.pyplot as plt
import numpy as np
import sys


from legacy_scripts.util_funcs import *




def test_infinite_ps(N = 32, save_gif = False):
    # Set up parameters for creating phase screens

    nx_size = N
    D = 1
    pxl_scale = D/nx_size
    r0 = 19e-2
    L0 = 10
    # wind_speed = 10 #m/s 
    # n_tests = 25 # 16
    # n_scrns = 100
    stencil_length_factor = 32
    phase_screen = PhaseScreenKolmogorov(nx_size, pxl_scale, r0, L0, stencil_length_factor=stencil_length_factor)
    # print(round(time.time()-s,2))
    plt.figure()
    plt.imshow(phase_screen.scrn)
    cbar = plt.colorbar()
    cbar.set_label('Wavefront deviation (radians)', labelpad=8)


    #save it
    frames = 5
    filenames = []

    for i in range(frames):
        # print(f"Calculating frame: {i+1} ........")
       
        phase_screen.add_row()
        # print(round(time.time()-s,2))
        plt.imshow(np.transpose(phase_screen.scrn))
        plt.draw()
        plt.pause(0.5)

        if save_gif:
            filename = f'frame_{i}.png'
            plt.savefig(filename)
            filenames.append(filename)  


    if save_gif:    
        with imageio.get_writer(f'anime{nx_size}by{nx_size}_pxl-scale={pxl_size}.gif', mode='I', duration=0.1) as writer:
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)

    # remove png files  
        for filename in set(filenames):
            os.remove(filename)


    
def test_multilayer_evol_angular_spec(N = 32, show_plot = True):
    wvl = 1550e-9
    delta1 = 10e-3
    deltan = 10e-3
    Dz = 1e3
    n = nscr = 5
    z = np.linspace(1,n, n) * Dz / n
    zt = [0, z]
    alpha = z / z[-1]
    r0 = 1
    delta = (1-alpha) * delta1 + alpha * deltan


    L0 = 10
    l0 = 0.01




    PS_objs = [PhaseScreenKolmogorov(N, delta[i], r0, L0, 32 )  for i in range(nscr)]
    
    proptor = propagator(N, wvl, delta1, deltan, z)
    frames = 20


    filenames = []
    frames_directory = os.path.join(os.path.dirname(__file__), 'imageFolder')

    if show_plot:
        plt.figure(figsize=(10,5))
    for i in range(frames):
    
        
        for phase_screen in PS_objs:
            phase_screen.add_row()
        PS_arr = [PS.scrn for PS in PS_objs]

        Uout, xn, yn = proptor.propagate(np.ones((N,N)), PS_arr)
        if show_plot:
            filename = os.path.join(frames_directory, f'frame_{i}.png')
        # print(round(time.time()-s,2))
        # write a side by side angle and intensity plot
        if show_plot:
            plt.subplot(1,2,1)
            plt.imshow(np.angle(Uout))
            plt.title(f'{N}x{N} phase screen evolution')
            plt.colorbar()
            plt.subplot(1,2,2)
            plt.imshow(np.abs(Uout)**2)
            plt.title(f'Intensity evolution')
            plt.colorbar()
            plt.draw()
            plt.pause(0.5)

        if show_plot:
            filename = os.path.join(frames_directory, f'frame_{i}.png')
        
            plt.savefig(filename)
            filenames.append(filename)
    if show_plot:
        gif_filename = os.path.join(os.path.dirname(__file__), 'imageFolder', f'full_prop_{N}x{N}.gif')

        with imageio.get_writer(gif_filename, mode='I', duration=0.001) as writer:
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)

        # remove png files  
        for filename in set(filenames):
            os.remove(filename)

    return Uout, xn, yn

    
def propWithOrWithoutEvol(N = 64, evol_steps = 0):
    wvl = 1550e-9
    delta1 = 10e-3
    deltan = 10e-3
    Dz = 1e3
    n = nscr = 5
    z = np.linspace(1,n, n) * Dz / n
    zt = [0, z]
    alpha = z / z[-1]
    r0 = 1
    delta = (1-alpha) * delta1 + alpha * deltan
     

    L0 = 10
    l0 = 0.01

    PS_objs = [PhaseScreenKolmogorov(N, delta[i], r0, L0, 32 )  for i in range(nscr)]
    
    proptor = propagator(N, wvl, delta1, deltan, z)


    # No evolution
    if evol_steps == 0:
        PS_objs = [PhaseScreenKolmogorov(N, delta[i], r0, L0, 32 ).scrn  for i in range(nscr)]

        Uout, xn, yn = proptor.propagate(np.ones((N,N)), PS_objs)
    else:
        PS_objs = [PhaseScreenKolmogorov(N, delta[i], r0, L0, 32 )  for i in range(nscr)]
        for i in range(evol_steps):
    
            
            for phase_screen in PS_objs:
                phase_screen.add_row()
            PS_arr = [PS.scrn for PS in PS_objs]

        Uout, xn, yn = proptor.propagate(np.ones((N,N)), PS_arr)
    return Uout, xn, yn

def validation_test(N = 64, evol_steps = 0, nreal = 40):
    deltan = 10e-3
    MCF2 = 0
    for i in tqdm(range(nreal)):
        Uout, xn, yn = propWithOrWithoutEvol(N, evol_steps)
        mask = circ(xn / .5, yn / .5, 1)
        MCF2 = MCF2 + corr2_ft(Uout, Uout, mask, deltan)

    MCDOC2 = np.abs(MCF2) / MCF2[N//2, N//2]

    x = np.linspace(0, N*deltan/2, N//2)
    u = np.exp(-3.44 * (x)**(5/3))
    plt.figure()
    plt.plot(MCDOC2[N//2, N//2:], label='Simulation')
    plt.plot(u, label='Theory')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Comparison of Simulation and Theory')
    plt.legend()
    plt.show()

def coherence_check_single_phase_screen(N = 64, nreal = 200):
    r0 = .4
    delta = 10e-3
    L0 = 1000
    l0 = 0.01


    MCF2 = 0
    x = np.arange(-N/2., N/2.)
    [nx, ny] = np.meshgrid(x, x)

    xn = nx * delta
    yn = ny * delta

    mask = circ(xn, yn , N * delta)
    
    MCF2 = 0
    for i in tqdm(range(nreal)):
        scrn = ft_sh_phase_screen(r0, N, delta, L0, l0)
        Uout = np.exp(1j*scrn)
        MCF2 = MCF2 + corr2_ft(Uout, Uout, mask, delta)

    MCDOC2 = np.abs(MCF2) / MCF2[N//2, N//2]

    x = np.linspace(0, N*delta/2, N//2)
    u = np.exp(-3.44 * (x/r0)**(5/3))
    plt.figure()
    plt.plot(x, MCDOC2[N//2, N//2:], label='Simulation')
    plt.plot(x, u, label='Theory')
    plt.xlabel('r')
    plt.ylabel('Value')
    plt.title('Comparison of Simulation and Theory')
    plt.legend()
    plt.show()

def coherence_check_extude_or_not(r0, delta, N = 64, nreal = 200, l0 = 0.01, L0 = 100, extude_levels = None, evaluate_over_evol = False):
    if extude_levels is None:
        x = np.arange(-N/2., N/2.)
        [nx, ny] = np.meshgrid(x, x)

        xn = nx * delta
        yn = ny * delta

        mask = circ(xn, yn , N * delta)
    
        MCF2 = 0
        for i in tqdm(range(nreal)):
            scrn = ft_sh_phase_screen(r0, N, delta, L0, l0)
            Uout = np.exp(1j*scrn)
            MCF2 = MCF2 + corr2_ft(Uout, Uout, mask, delta)

        MCDOC2 = np.abs(MCF2) / MCF2[N//2, N//2]

        x = np.linspace(0, N*delta/2, N//2)
        u = np.exp(-3.44 * (x/r0)**(5/3))
        plt.figure()
        plt.plot(x, MCDOC2[N//2, N//2:], label='Simulation')
        plt.plot(x, u, label='Theory')
        plt.xlabel('r')
        plt.ylabel('Value')
        plt.title(f'Comparison of Simulation and Theory - r0={r0}, delta={delta}, N={N}')
        plt.legend()
        #Save the plot in imageFolder/validation
        filename = os.path.join(os.path.dirname(__file__), 'imageFolder\Validation', f'validation_r0={r0}_delta={delta}_N={N}-NoExtrude.png')
        plt.savefig(filename)
        plt.show()
    else:
        x = np.arange(-N/2., N/2.)
        [nx, ny] = np.meshgrid(x, x)

        xn = nx * delta
        yn = ny * delta

        mask = circ(xn, yn , N * delta)
    
        MCF2 = 0
        PS_object = PhaseScreenKolmogorov(N, delta, r0, L0, 32 )
        if evaluate_over_evol:
            for i in range(extude_levels):
                PS_object.add_row()
                Uout = np.exp(1j*PS_object.scrn)
                MCF2 = MCF2 + corr2_ft(Uout, Uout, mask, delta)
            MCDOC2 = np.abs(MCF2) / MCF2[N//2, N//2]
            
            x = np.linspace(0, N*delta/2, N//2)
            u = np.exp(-3.44 * (x/r0)**(5/3))
            plt.figure()
            plt.plot(x, MCDOC2[N//2, N//2:], label='Simulation')
            plt.plot(x, u, label='Theory')
            plt.xlabel('r')
            plt.ylabel('Value')
            plt.title(f'Comparison of Simulation and Theory - r0={r0}, delta={delta}, N={N}')
            plt.legend()
            #Save the plot in imageFolder/validation
            filename = os.path.join(os.path.dirname(__file__), 'imageFolder\Validation', f'validation_r0={r0}_delta={delta}_N={N}_evalOverExtursionOf{extude_levels}.png')
            plt.savefig(filename)
            plt.show()
        else:
            x = np.arange(-N/2., N/2.)
            [nx, ny] = np.meshgrid(x, x)

            xn = nx * delta
            yn = ny * delta

            mask = circ(xn, yn , N * delta)
        
            MCF2 = 0
            PS_object = PhaseScreenKolmogorov(N, delta, r0, L0, 32 )
            for i in tqdm(range(nreal)):
                for i in range(extude_levels):
                    PS_object.add_row()
                Uout = np.exp(1j*PS_object.scrn)
                MCF2 = MCF2 + corr2_ft(Uout, Uout, mask, delta)
            MCDOC2 = np.abs(MCF2) / MCF2[N//2, N//2]
            
            x = np.linspace(0, N*delta/2, N//2)
            u = np.exp(-3.44 * (x/r0)**(5/3))
            plt.figure()
            plt.plot(x, MCDOC2[N//2, N//2:], label='Simulation')
            plt.plot(x, u, label='Theory')
            plt.xlabel('r')
            plt.ylabel('Value')
            plt.title(f'Comparison of Simulation and Theory - r0={r0}, delta={delta}, N={N}')
            plt.legend()

            #Save the plot in imageFolder/validation
            filename = os.path.join(os.path.dirname(__file__), 'imageFolder\Validation', f'validation_r0={r0}_delta={delta}_N={N}_Extruded&NRealization.png')
            plt.savefig(filename)

            plt.show()

def main():
    # coherence_check_single_phase_screen(128)

    
    
    coherence_check_extude_or_not(1, 5e-2, 128, L0=100,extude_levels=2000 ,evaluate_over_evol=True)
    # test_infinite_ps(64)  
    

    # validation_test(64, 10, 10) 
    # propWithOrWithoutEvol(64, 10) 

   
        

    

if __name__ == "__main__":
    main()