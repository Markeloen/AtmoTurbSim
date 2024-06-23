import numpy as np # type: ignore
import matplotlib.pyplot as plt

from phase_screens.infinitephasescreen import PhaseScreenVonKarman
from phase_screens.phasescreen_github import *
from propagation.basic_funcs import *

# # Obsolete !!!!!
# def create_ax_fig_phase_screen(input_data):
#     fig, ax = plt.subplots()
#     cax = ax.imshow(input_data, cmap='viridis')
#     cbar = fig.colorbar(cax, ax=ax)
#     cbar.set_label('Intensity')
    
#     ax.set_title('Phase Screen Data Visualization')
#     ax.set_xlabel('X Coordinate')
#     ax.set_ylabel('Y Coordinate')
    
#     return fig, ax

# def create_ax_fig_coherence_validation(x_axis, theoretical_, MDOC2):
#     fig, ax = plt.subplots()
    
#     N = len(MDOC2)
    
#     ax.plot(x_axis, theoretical_, label='Theoretical Modulus of Coherence Function')
#     ax.plot(x_axis, MDOC2[N//2, N//2:], label='Modulus of Coherence Function')
#     ax.set_title('Modulus of Coherence Function Validation')
#     ax.set_xlabel('Distance (m)')
#     ax.set_ylabel('Modulus of Coherence Function')
#     ax.legend()
#     return fig, ax


class Layer:
    def __init__(self,nx_size, pixel_scale, r0, L0 = 50) -> None:
        self.nx_size = nx_size
        self.pixel_scale = pixel_scale
        self._phase_screen_object = PhaseScreenVonKarman(nx_size, pixel_scale, r0, L0)
        self.MCF2 = 0
        self.r0 = r0
        self.x = np.linspace(0, self.nx_size*self.pixel_scale/2, self.nx_size//2, self.nx_size//2)
        self.theoretical_mod_function = 0
        self.phase_screen_plot_fig = None
        self.phase_screen_plot_ax = None
        self.theoretical_mod_function =  np.exp(-3.44 * (self.x/self.r0)**(5/3))
        
   
        
        
    def extude_return_scrn(self):
        self._phase_screen_object.add_row()
        self.update_mutual_coherence()
   
        return self._phase_screen_object.scrn

    def extude_return_column(self):
        self._phase_screen_object.add_row()
        self.update_mutual_coherence()
        
        return self._phase_screen_object.scrn[:, -1]
    
    def update_mutual_coherence(self):
        
        x = np.arange(-self.nx_size/2., self.nx_size/2.)
        [nx, ny] = np.meshgrid(x, x)

        xn = nx * self.pixel_scale
        yn = ny * self.pixel_scale

        mask = circ(xn, yn , self.nx_size * self.pixel_scale)
        
        Uout = np.exp(1j*self._phase_screen_object.scrn)
        self.MCF2 = self.MCF2 + corr2_ft(Uout, Uout, mask, self.pixel_scale)
    
    def return_MDOC2(self):
        return (np.abs(self.MCF2) / self.MCF2[self.nx_size//2, self.nx_size//2])[self.nx_size//2, self.nx_size//2:]
    
    def retrun_modulus_function_xaxis(self):
        return self.return_MDOC2(), self.theoretical_mod_function, self.x
        
    
    @property
    def scrn(self):
        return self._phase_screen_object.scrn
