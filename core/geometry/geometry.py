import numpy as np
from scipy import integrate, optimize
from math import cos
from dataclasses import dataclass, field

from core.layer.layer import Layer
from core.utils import calc_r0_profile, calculate_number_of_extrusions

@dataclass
class Geometry:
    """
    Represents the geometry and dynamics of a multi-layered phase screen simulation.
    """
    nx_size: int
    pixel_scale: float
    whole_simulation_time: float = 1  # seconds
    per_tick_simulation: float = 1e-3  # seconds
    number_of_layers: int = 10
    satellite_orbit: float = 6e5
    ground_wind_speed: float = 5
    r0_array: list = field(default_factory=list, init=False)
    layer_height_array: list = field(default_factory=list, init=False)
    wind_profile_array: list = field(default_factory=list, init=False)
    layer_object_array: list = field(default_factory=list, init=False)
    number_extrusions_array_per_tick: list = field(default_factory=list, init=False)
    number_extrusion_array_whole: list = field(default_factory=list, init=False)

    def __post_init__(self):
        """
        Initializes the Geometry object, calculating necessary profiles and creating Layer objects.
        """
        # Calculate r0, layer height, and wind profile
        results = calc_r0_profile(self.ground_wind_speed, self.satellite_orbit, True)
        self.r0_array, self.layer_height_array, self.wind_profile_array = results

        # Create Layer objects and number of extrusions array
        for i in range(self.number_of_layers):
            self.layer_object_array.append(Layer(self.nx_size, self.pixel_scale, self.r0_array[i]))
            print(f"Currently creating a Layer at height: {self.layer_height_array[i]} ...")
            # Calculate phase screen evolution speed at each layer (m/s)
            self.number_extrusions_array_per_tick.append(
                calculate_number_of_extrusions(self.layer_object_array[i], self.wind_profile_array[i], self.per_tick_simulation)
            )

        self.number_extrusion_array_whole = np.ceil(
            np.array(self.number_extrusions_array_per_tick) * self.whole_simulation_time / self.per_tick_simulation
        )

    def move_one_tick_all_layers(self):
        """
        Moves all layers for one tick and returns phase screens as an array.
        
        :return: Array of phase screens for each layer
        """
        ps_arr = []
        extr_per_tick = np.ceil([3 * item for item in self.number_extrusions_array_per_tick])

        for indx, ext_lvl in enumerate(extr_per_tick):
            for _ in range(int(ext_lvl)):
                self.layer_object_array[indx].extrude_return_scrn()
            ps_arr.append(self.layer_object_array[indx].scrn)

        return ps_arr

    def show_object_info(self):
        """
        Prints the current configuration and state of the Geometry object.
        """
        print(f"nx_size: {self.nx_size}")
        print(f"pixel_scale: {self.pixel_scale}")
        print(f"per_tick_simulation: {self.per_tick_simulation}")
        print(f"number_of_layers: {self.number_of_layers}")
        print(f"satellite_orbit: {self.satellite_orbit}")
        print(f"ground_wind_speed: {self.ground_wind_speed}")
        print(f"r0_array: {self.r0_array}")
        print(f"layer_height_array: {self.layer_height_array}")
        print(f"wind_profile_array: {self.wind_profile_array}")
        print(f"layer_object_array: {self.layer_object_array}")
        print(f"number_extrusions_array_per_tick: {self.number_extrusions_array_per_tick}")
        print(f"number_extrusion_array_whole: {self.number_extrusion_array_whole}")

if __name__ == "__main__":
    geometry = Geometry(128, 2e-2)
    geometry.show_object_info()
