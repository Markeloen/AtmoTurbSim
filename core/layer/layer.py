import numpy as np
import matplotlib.pyplot as plt

from core.phase_screens import PhaseScreenVonKarman
from core.phase_screens.basic_funcs import circ, corr2_ft



class Layer:
    """
    Represents a layer with a phase screen object, and handles operations related to phase screens and mutual coherence.
    """
    def __init__(self, nx_size: int, pixel_scale: float, r0: float, L0: float = 20.0) -> None:
        """
        Initializes the Layer with the given parameters.
        
        :param nx_size: Size of the grid in pixels
        :param pixel_scale: Scale of each pixel
        :param r0: Fried parameter
        :param L0: Outer scale of turbulence (default is 20)
        """
        self.nx_size = nx_size
        self.pixel_scale = pixel_scale
        self.r0 = r0
        self._phase_screen_object = PhaseScreenVonKarman(nx_size, pixel_scale, r0, L0)
        self.MCF2 = 0
        self.x = np.linspace(0, self.nx_size * self.pixel_scale / 2, self.nx_size // 2)
        self.theoretical_mod_function = np.exp(-3.44 * (self.x / self.r0) ** (5 / 3))
        self.phase_screen_plot_fig = None
        self.phase_screen_plot_ax = None

    def extrude_return_scrn(self) -> np.ndarray:
        """
        Adds a new row to the phase screen and updates mutual coherence.
        
        :return: Updated phase screen
        """
        self._phase_screen_object.add_row()
        self.update_mutual_coherence()
        return self._phase_screen_object.scrn

    def extrude_return_column(self) -> np.ndarray:
        """
        Adds a new row to the phase screen and updates mutual coherence, then returns the last column of the screen.
        
        :return: Last column of the updated phase screen
        """
        self._phase_screen_object.add_row()
        self.update_mutual_coherence()
        return self._phase_screen_object.scrn[:, -1]

    def update_mutual_coherence(self) -> None:
        """
        Updates the mutual coherence function (MCF) using the current phase screen.
        """
        x = np.arange(-self.nx_size / 2., self.nx_size / 2.)
        nx, ny = np.meshgrid(x, x)
        xn = nx * self.pixel_scale
        yn = ny * self.pixel_scale
        mask = circ(xn, yn, self.nx_size * self.pixel_scale)
        Uout = np.exp(1j * self._phase_screen_object.scrn)
        self.MCF2 += corr2_ft(Uout, Uout, mask, self.pixel_scale)

    def return_MDOC2(self) -> np.ndarray:
        """
        Returns the normalized magnitude of the mutual coherence function.
        
        :return: Normalized MCF magnitude
        """
        return (np.abs(self.MCF2) / self.MCF2[self.nx_size // 2, self.nx_size // 2])[self.nx_size // 2, self.nx_size // 2:]

    def return_modulus_function_xaxis(self) -> tuple:
        """
        Returns the modulus function along the x-axis.
        
        :return: Tuple containing the MCF, theoretical modulation function, and x-axis values
        """
        return self.return_MDOC2(), self.theoretical_mod_function, self.x

    @property
    def scrn(self) -> np.ndarray:
        """
        Property to get the current phase screen.
        
        :return: Current phase screen
        """
        return self._phase_screen_object.scrn
