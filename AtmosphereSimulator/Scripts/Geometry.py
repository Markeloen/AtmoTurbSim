import numpy as np # type: ignore
import scipy.integrate as integrate # type: ignore
from scipy import optimize # type: ignore
from math import cos
from dataclasses import dataclass, field

from Scripts.Layer import Layer
from Scripts.Utils import *



@dataclass
class Geometry:
    nx_size: int
    pixel_scale: float
    per_tick_simulation: float = 1e-3
    number_of_layers: int = 10
    satellite_orbit: float = 6e5
    ground_wind_speed: float = 5
    r0_array: list = field(default_factory=list, init=False)
    layer_height_array: list = field(default_factory=list, init=False)
    wind_proofile_array: list = field(default_factory=list, init=False)
    layer_object_array: list = field(default_factory=list, init=False)
    number_of_calculated_extrusions_array: list = field(default_factory=list, init=False)
    
    
    

    def __post_init__(self):
        # Calculating r0, layer height and wind profile
        results = calc_r0_profile(self.ground_wind_speed, self.satellite_orbit, True)
        self.r0_array, self.layer_height_array, self.wind_proofile_array = results
        for i in range(len(self.r0_array)):
            self.layer_object_array.append(Layer(self.nx_size, self.pixel_scale, self.r0_array[i]))

        # Creating Lyaer objects
        # Creating number of extrusions array
        for i in range(self.number_of_layers):
            self.layer_object_array.append(Layer(self.nx_size, self.pixel_scale, self.r0_array[i]))
            print(f"Currently Creating a Lyaer at height: {self.layer_height_array[i]} ...")
            # calculating phase screen evolution speed at each layer (m/s)
            # Double check this!!! V(h) is passeed as the layer's speed -> contains wind speed and satellite speed combined
            self.number_of_calculated_extrusions_array.append(calculate_number_of_extrusions(self.layer_object_array[i], self.wind_proofile_array[i],
                                                                                             self.per_tick_simulation))

        print(f"Number of extrusions: {self.number_of_calculated_extrusions_array}")
if __name__ == "__main__":
    geometry = Geometry(128, 2e-2)