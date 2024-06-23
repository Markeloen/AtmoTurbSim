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
    whole_simulation_time: float = 1 # seconds
    per_tick_simulation: float = 1e-3 # seconds
    number_of_layers: int = 10
    satellite_orbit: float = 6e5
    ground_wind_speed: float = 5
    r0_array: list = field(default_factory=list, init=False)
    layer_height_array: list = field(default_factory=list, init=False)
    wind_proofile_array: list = field(default_factory=list, init=False)
    layer_object_array: list = field(default_factory=list, init=False)
    number_extrusions_array_per_tick: list = field(default_factory=list, init=False)
    number_extrusion_array_whole: list = field(default_factory=list, init=False)
    

    
    
    def __post_init__(self):
        # Calculating r0, layer height and wind profile
        results = calc_r0_profile(self.ground_wind_speed, self.satellite_orbit, True)
        self.r0_array, self.layer_height_array, self.wind_proofile_array = results
        
        # Creating Lyaer objects
        # Creating number of extrusions array
        for i in range(self.number_of_layers):
            self.layer_object_array.append(Layer(self.nx_size, self.pixel_scale, self.r0_array[i]))
            print(f"Currently Creating a Lyaer at height: {self.layer_height_array[i]} ...")
            # calculating phase screen evolution speed at each layer (m/s)
            # Double check this!!! V(h) is passeed as the layer's speed -> contains wind speed and satellite speed combined
            self.number_extrusions_array_per_tick.append(calculate_number_of_extrusions(self.layer_object_array[i], self.wind_proofile_array[i],
                                                                                             self.per_tick_simulation))

        self.number_extrusion_array_whole = np.array(self.number_extrusions_array_per_tick) * self.whole_simulation_time / self.per_tick_simulation
        self.number_extrusion_array_whole = np.ceil(self.number_extrusion_array_whole)
        
    def show_object_info(self):
        print(f"nx_size: {self.nx_size}")
        print(f"pixel_scale: {self.pixel_scale}")
        print(f"per_tick_simulation: {self.per_tick_simulation}")
        print(f"number_of_layers: {self.number_of_layers}")
        print(f"satellite_orbit: {self.satellite_orbit}")
        print(f"ground_wind_speed: {self.ground_wind_speed}")
        print(f"r0_array: {self.r0_array}")
        print(f"layer_height_array: {self.layer_height_array}")
        print(f"wind_proofile_array: {self.wind_proofile_array}")
        print(f"layer_object_array: {self.layer_object_array}")
        print(f"number_extrusions_array_per_tick: {self.number_extrusions_array_per_tick}")
        print(f"number_extrusion_array_whole: {self.number_extrusion_array_whole}")
        
        
if __name__ == "__main__":
    geometry = Geometry(128, 2e-2)
    geometry.show_object_info()