import numpy as np # type: ignore


from phase_screens.infinitephasescreen import PhaseScreenVonKarman
from phase_screens.phasescreen_github import *



class Layer:
    def __init__(self,nx_size, pixel_scale, r0, L0 = 50) -> None:
        self._phase_screen_object = PhaseScreenVonKarman(nx_size, pixel_scale, r0, L0)

    def extude_return_scrn(self):
        self._phase_screen_object.add_row()

    def extude_return_column(self):
        self._phase_screen_object.add_row()
        return self._phase_screen_object.scrn[:, -1]
    
    @property
    def scrn(self):
        return self._phase_screen_object.scrn
