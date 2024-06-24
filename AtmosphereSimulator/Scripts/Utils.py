import numpy as np # type: ignore
import scipy.integrate as integrate # type: ignore
from scipy import optimize # type: ignore
from math import cos
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from PIL import Image
import io

from Scripts.Layer import Layer
# from Scripts.Geometry import Geometry


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
        #Calculating satellite slew rate, v_tang / range = slew rate -> rad/s
        ws = circ_orbit_geo(stellite_orbit_height)[1] / circ_orbit_geo(stellite_orbit_height)[0]

        #Creating Bufton wind model function - for wind speed only
        bufton_function = lambda h: ws * h + ground_wind_speed + 30 * np.exp(-((h - 9400) / 4800) ** 2)
        # bufton_function for Cn2 use , without the term Ws*h, why? cause satellite speed is not relevant to Cn2 :)
        bufton_function_Cn2 = lambda h: ground_wind_speed + 30 * np.exp(-((h - 9400) / 4800) ** 2)

        #calculing rms wind
        rms_wind = np.sqrt(1 / (15 * 1e3) * integrate.quad(bufton_function_Cn2, 5e3, 20e3)[0])

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
        for i in range(len(h_arr)-1):
            r0_array.append(r0_calc(h_arr[i], h_arr[i+1]))
        h_arr = h_arr[1:]
        if print_results:
            print(f"r0_array: {r0_array}")

        wind_profile = list(map(bufton_function, h_arr))
        if print_results:
            print(f"Wind profile: {wind_profile}")

        

        if print_results:
            print(f"Number of layers : {len(h_arr)}")
            print(h_arr)
        # doing the next two lines so every element in the array is the same
        
        h_arr  = [item.item() if isinstance(item, np.ndarray) else item for item in h_arr]
        wind_profile = [item.item() if isinstance(item, np.ndarray) else item for item in wind_profile]
        # wind_profile = [item[0] for item in wind_profile]
        return r0_array, h_arr, wind_profile
        
def calculate_number_of_extrusions(layer: Layer, v_total, simulation_tick_time_sec):
    """Bufton model V(h) gives out the evolution speed of a phase screen at height h = v_total"""
    
    delta_x = v_total * simulation_tick_time_sec
    columns_per_tick = delta_x / layer.pixel_scale

    return columns_per_tick


# Incomplete!!
def show_geomerty_layers(gerometry, shown_layer_array = [0, 4, 9]):
    """ This function will create a grid view plot to show phase screen 
    evolution of the layers in the geometry object"""
    
    n = len(shown_layer_array)
    fig, axes = plt.subplots(n, 2, figsize=(10, 6))
    
    gerometry.show_object_info()
    
    # Configure font size for ticks
    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['ytick.labelsize'] = 8
    
    r0 = 20e-2
    layer = Layer(128, 2e-2, r0)
    # Extract data
    layer.extude_return_column()
    data_xaxis = layer.retrun_modulus_function_xaxis()
    
    images = []  # List to hold images
        
    for i in range(n):
    # Display the phase screen with a more colorful and clear colormap
        im = axes[i, 0].imshow(layer.scrn, cmap='viridis')
        fig.colorbar(im, ax=axes[i, 0], orientation='vertical', shrink=0.6)
        axes[i, 0].set_title('Phase Screen')
        
        # Optionally, remove tick labels for a cleaner look
        axes[i, 0].set_xticklabels([])
        axes[i, 0].set_yticklabels([])

        # Plot modulus function with clearly distinguished line styles and markers
        
        axes[i, 1].plot(data_xaxis[2], data_xaxis[0], 'r-', label='Modulus 1')
        
        axes[i, 1].autoscale(tight=False)
        axes[i, 1].plot(data_xaxis[2], data_xaxis[1], 'b--', label='Modulus 2')
        axes[i, 1].legend(loc='upper right')
        axes[i, 1].set_xlabel('X-axis')
        axes[i, 1].set_ylabel('Modulus Value')
        
        
    # Improve overall aesthetics
    plt.tight_layout()
    plt.show(block=False)

    counter = 0
    for idx, layer_ith in enumerate(shown_layer_array):
        for ext_level in range(int(gerometry.number_extrusion_array_whole[layer_ith])):
            gerometry.layer_object_array[layer_ith].extude_return_scrn()
            
                # Update data
            data_xaxis = gerometry.layer_object_array[layer_ith].retrun_modulus_function_xaxis()
            
            # Update phase screen
            axes[idx, 0].images[0].set_data(gerometry.layer_object_array[layer_ith].scrn)
            axes[idx, 0]
            # Optionally, remove tick labels to keep the updated phase screen clean
            axes[idx, 0].set_xticklabels([])
            axes[idx, 0].set_yticklabels([])
            axes[idx, 1].set_title(f'Modulus Function for Layer {layer_ith} iteration {ext_level}/{gerometry.number_extrusion_array_whole[layer_ith]}')
            # Update modulus function plots
            axes[idx, 1].lines[0].set_ydata(data_xaxis[0])
            axes[idx, 1].lines[0].set_xdata(data_xaxis[2])
            axes[idx, 1].lines[1].set_ydata(np.abs(data_xaxis[1]))
            axes[idx, 1].lines[1].set_xdata(data_xaxis[2])
        
            if gerometry.number_extrusion_array_whole[layer_ith] < 1000:
                plt.pause(0.01)
            plt.pause(0.001)
            
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=80)
            buf.seek(0)
            img = Image.open(buf)
            img.load()
            images.append(img.copy())
            buf.close()
            counter += 1
            if counter > 3000:
                break
    
    # Save images as a GIF
    images[0].save('phase_screens.gif', save_all=True, append_images=images[1:], optimize=False, duration=40, loop=0)


    
    # Improve overall aesthetics
    plt.tight_layout()
    plt.show(block=False)
    
    return fig, axes





    
    
if __name__ == "__main__":
    pass