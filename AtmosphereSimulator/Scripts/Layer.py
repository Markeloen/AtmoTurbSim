import numpy as np # type: ignore
import matplotlib.pyplot as plt

from phase_screens.infinitephasescreen import PhaseScreenVonKarman
from phase_screens.phasescreen_github import *
from propagation.basic_funcs import *


def create_ax_fig_2D(input_data):
    fig, ax = plt.subplots()
    cax = ax.imshow(input_data, cmap='viridis')
    cbar = fig.colorbar(cax, ax=ax)
    cbar.set_label('Intensity')
    
    ax.set_title('Phase Screen Data Visualization')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    
    return fig, ax



class Layer:
    def __init__(self,nx_size, pixel_scale, r0, L0 = 50) -> None:
        self.nx_size = nx_size
        self.pixel_scale = pixel_scale
        self._phase_screen_object = PhaseScreenVonKarman(nx_size, pixel_scale, r0, L0)
        self.MCF2 = 0
    
    def __post_init__(self):
        self.phase_screen_plot_fig, self.phase_screen_plot_ax = create_ax_fig_2D(self._phase_screen_object.scrn)
        self.coherence_length_plot_fig, self.coherence_length_plot_ax = create_ax_fig_2D(self._phase_screen_object.r0)

    def extude_return_scrn(self):
        self._phase_screen_object.add_row()
        self.update_mutual_coherence()

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
        return np.abs(self.MCF2) / self.MCF2[self.N//2, self.N//2]
    
    @property
    def scrn(self):
        return self._phase_screen_object.scrn
