import numpy as np # type: ignore
import scipy.integrate as integrate # type: ignore
from scipy import optimize # type: ignore
from math import cos
from dataclasses import dataclass, field

from Scripts.Layer import Layer


def circ_orbit_geo(h_orbit, elev_angle_deg = 90):
    # Constants
    c = 299792458  # Speed of light in meters per second
    G = 6.67430e-11  # Gravitational constant in m^3 kg^-1 s^-2
    M = 5.972e24  # Mass of Earth in kilograms
    r_earth = 6.378e6  # Radius of Earth in meters
    
    # Calculating the radius from the center of Earth to the satellite
    R = r_earth + h_orbit
    
    # Calculate orbital velocity for a perfect circular orbit
    v_orbit = np.sqrt(G * M / R)
    
    # Convert elevation angle to radians
    gamma_horiz_rad = np.deg2rad(elev_angle_deg)
    
    # Solve the quadratic equation to find the line of sight distance L
    a = 1
    b = -2 * r_earth * np.cos(gamma_horiz_rad + np.pi/2)
    c = r_earth**2 - R**2
    coeffs = [a, b, c]
    L = max(np.roots(coeffs))
    
    # Angle from the center of the Earth to the line of sight
    gamma_earth_rad = np.arccos((L**2 - R**2 - r_earth**2) / (-2 * R * r_earth))
    
    # Calculate range using the law of cosines
    range_km = np.sqrt(R**2 + r_earth**2 - 2 * R * r_earth * np.cos(gamma_earth_rad))
    
    # Calculate the components of velocity
    psi = np.pi - abs(gamma_earth_rad) - (np.pi/2 + gamma_horiz_rad)
    v_radial = v_orbit * np.cos(np.pi/2 - psi)
    v_tangential = v_orbit * np.sin(np.pi/2 - psi)
    
    return range_km, v_tangential, v_radial



def calc_r0_profile(ground_wind_speed, stellite_orbit_height, print_results = False, lamda = 1550e-9, A = 1.7e-14):
        
        k = 2 * np.pi / lamda


        # !!!! Ask Michael
        #Calculating satellite slew rate
        ws = circ_orbit_geo(stellite_orbit_height)[2]

        #Creating Bufton wind model function
        bufton_function = lambda h: ws * h + ground_wind_speed + 30 * np.exp(-((h - 9400) / 4800) ** 2)

        #calculing rms wind
        rms_wind = np.sqrt(1 / (15 * 1e3) * integrate.quad(bufton_function, 5e3, 20e3)[0])

        # cn2 calculation
        cn2 = lambda h : (.00594 * (rms_wind / 27) ** 2 * (1e-5*h) ** 10 * np.exp(-h/1000) + 2.7e-16 * np.exp(-h/1500) + A * np.exp(-h/100))


        # r0 calculation
        r0_calc = lambda h1, h2 : ( .423 * k ** 2 * integrate.quad(cn2, h1, h2)[0] ) ** (-3/5)

        
        #scintillation index calculation
        etha = 0
        scintillation_index = lambda h0, H : 2.25 * k ** (7/6) * (1/cos(etha)) ** (11/6) * integrate.quad(lambda h : cn2(h) * (h-h0) ** (5/6), h0, H)[0]

        # 15-layer setup
        setup_h = [1e2, 1e3, 2e3, 3e3, 4e3, 5e3, 6e3, 7e3, 8e3, 9e3, 10e3, 12e3, 14e3, 16e3, 18e3, 20e3]

        # sci_indx < 0.1 sci_indx(full_path)

        threshold = 0.1 * integrate.quad(lambda h : cn2(h) * (h-0) ** (5/6), 0, 20000)[0]

        if print_results:
            print(f"Threshold: {threshold}")

        H = 500

        equation_to_solve = lambda h0 : integrate.quad(lambda h : cn2(h) * (h) ** (5/6), h0, H)[0] - threshold


        # equation_to_solve = lambda H : integrate.quad(lambda h : cn2(h) * (h-h0) ** (5/6), h0, H)[0] - threshold
        initial_guess = 300

        result = optimize.fsolve(equation_to_solve, initial_guess)

        # result


        H = result = 0
        h_arr = [H]
        while(result <= 20000):
            h0 = result
            initial_guess = h0
            equation_to_solve = lambda H : integrate.quad(lambda h : cn2(h) * (h) ** (5/6), h0, H)[0] - threshold
            result = optimize.fsolve(equation_to_solve, initial_guess)
            # if print_results:
            #     print(f" loop: {integrate.quad(lambda h : cn2(h) * (h) ** (5/6), h0, result)[0]}")
            #     print(result)
            h_arr.append(result)

        

        


        r0_calc = lambda h1, h2 : ( .423 * k ** 2 * integrate.quad(cn2, h1, h2)[0] ) ** (-3/5)

        r0_array = []
        # Is this correct?
        for i in range(len(h_arr)-2):
            r0_array.append(r0_calc(h_arr[i], h_arr[i+1]))

        if print_results:
            print(f"r0_array: {r0_array}")

        wind_profile = list(map(bufton_function, h_arr))
        if print_results:
            print(f"Wind profile: {wind_profile}")

        h_arr = h_arr[1:]

        if print_results:
            print(f"Number of layers : {len(h_arr)}")
            print(h_arr)
        # doing the next two lines so every element in the array is the same
        h_arr = [item[0] for item in h_arr]
        wind_profile = [item[0] for item in wind_profile]
        return r0_array, h_arr, wind_profile
        
def calculate_number_of_extrusions(layer: Layer, wind_speed_rms, layer_rms_speed, simulation_tick_time_sec):
    v_total = layer_rms_speed + wind_speed_rms
    delta_x = v_total * simulation_tick_time_sec
    delta_x_each_screen = layer.pixel_scale * layer.screen_size
    columns_per_tick = delta_x / delta_x_each_screen

    return columns_per_tick