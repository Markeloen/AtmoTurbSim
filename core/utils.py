import numpy as np # type: ignore
import scipy.integrate as integrate # type: ignore
from scipy import optimize # type: ignore
from math import cos
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from PIL import Image
import io
import tensorflow as tf
import numpy as np
import scipy
from scipy.integrate import quad
from scipy.optimize import fsolve
from datetime import datetime
import os



from core.layer import Layer
# from Scripts.Geometry import Geometry

import numpy as np
from scipy.integrate import quad
from scipy.optimize import fsolve
from typing import Callable

def solve_for_A(k: float, w: float, delta_z: float, r0_pw_given: float) -> float:
    """
    Solve for the parameter A in the integral equation using the given V(h) function.

    Parameters:
    k (float): Constant k.
    w (float): RMS wind speed.
    delta_z_km (float): The range of z in kilometers.
    r0_pw_given (float): The given value of r0,pw.
    V_function (Callable[[float], float]): Function that calculates V(h) given h.

    Returns:
    float: The value of A that satisfies the equation.
    """
 
 
    

    # Define Cn^2(h)
    def Cn2(h: float, A: float) -> float:
        term1 = 0.00594 * (w / 27)**2 * (10**-5 * h)**10 * np.exp(-h / 1000)
        term2 = 2.7 * 10**-16 * np.exp(-h / 1500)
        term3 = A * np.exp(-h / 100)
        return term1 + term2 + term3

    # Define the integral function
    def integral_function(A: float) -> float:
        integrand = lambda z: Cn2(z, A)
        integral_value, _ = quad(integrand, 0, delta_z)
        return integral_value

    # Define the equation to solve for A
    def equation_to_solve(A: float) -> float:
        integral_value = integral_function(A)
        r0_pw_calculated = (0.423 * k**2 * integral_value)**(-3/5)
        return r0_pw_calculated - r0_pw_given

    # Solve for A
    A_initial_guess = 1.7e-14  # Initial guess for A
    A_solution = fsolve(equation_to_solve, A_initial_guess)

    return A_solution[0]



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



def calc_r0_profile(ground_wind_speed, stellite_orbit_height, print_results = False, r0 = None, num_layers = 20, lamda = 1550e-9, A = 1.7e-14, max_altitude = 20e3):
        
        k = 2 * np.pi / lamda
        
        

    
        
        #Calculating satellite slew rate, v_tang / range = slew rate -> rad/s
        ws = circ_orbit_geo(stellite_orbit_height)[1] / circ_orbit_geo(stellite_orbit_height)[0]

        #Creating Bufton wind model function - for wind speed only
        bufton_function = lambda h: ws * h + ground_wind_speed + 30 * np.exp(-((h - 9400) / 4800) ** 2)
        # bufton_function for Cn2 use , without the term Ws*h, why? cause satellite speed is not relevant to Cn2 :)
        bufton_function_Cn2 = lambda h: ground_wind_speed + 30 * np.exp(-((h - 9400) / 4800) ** 2)

        #calculing rms wind
        rms_wind = np.sqrt(1 / (15 * 1e3) * integrate.quad(bufton_function_Cn2, 5e3, max_altitude)[0])
        
        if r0 != None:
            A = solve_for_A(k, rms_wind, max_altitude, r0)

        # cn2 calculation
        cn2 = lambda h : (.00594 * (rms_wind / 27) ** 2 * (1e-5*h) ** 10 * np.exp(-h/1000) + 2.7e-16 * np.exp(-h/1500) + A * np.exp(-h/100))
        

        # r0 calculation
        r0_calc = lambda h1, h2 : ( .423 * k ** 2 * integrate.quad(cn2, h1, h2)[0] ) ** (-3/5)
        
        
        #scintillation index calculation
        etha = 0
        scintillation_index = lambda h0, H : 2.25 * k ** (7/6) * (1/cos(etha)) ** (11/6) * integrate.quad(lambda h : cn2(h) * (h-h0) ** (5/6), h0, H)[0]

        

        # sci_indx < (num_layers / 100) * sci_indx(full_path)
        
        

        threshold = (1 / num_layers) * integrate.quad(lambda h : cn2(h) * (h-0) ** (5/6), 0, max_altitude)[0]

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
        while(result <= max_altitude):
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


# Complete!!
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

def rect(x,width):
    return np.heaviside(x+width/2,0.5) - np.heaviside(x-width/2,0.5)

def circ(x,y,D):
    r = np.sqrt(x**2 + y**2)
    return (r < D/2.0).astype(np.float32)

def ift2(G, delta_f, FFT=None):
    """
    Wrapper for inverse fourier transform

    Parameters:
        G: data to transform
        delta_f: pixel seperation
        FFT (FFT object, optional): An accelerated FFT object
    """

    N = G.shape[0]

    if FFT:
        g = np.fft.fftshift(FFT(np.fft.fftshift(G))) * (N * delta_f) ** 2
    else:
        g = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(G))) * (N * delta_f) ** 2

    return g

def ft2(data, delta):
    """
    A properly scaled 1-D FFT
    Parameters:
        data (ndarray): An array on which to perform the FFT
        delta (float): Spacing between elements
    Returns:
        ndarray: scaled FFT
    """
    DATA = np.fft.fftshift(
            np.fft.fft2(np.fft.fftshift(data))) * delta**2
    return DATA

def tf_ft2(g, delta):
    g = tf.cast(g, tf.complex64)
    delta = tf.complex(tf.cast(delta, dtype=tf.float32), 0.0)
    return tf.signal.fftshift(tf.signal.fft2d(tf.signal.fftshift(g))) * delta**2




def tf_ift2(g, delta_f):
    N = np.size(g,0) # assume square
    g = tf.cast(g, tf.complex64)
    delta_f = tf.complex(tf.cast(delta_f, dtype=tf.float32), 0.0)
    return tf.signal.ifftshift(tf.signal.ifft2d(tf.signal.ifftshift(g))) * (N * delta_f)**2

# Dependency issue
# def fresnel_prop_square_ap(x2, y2, D1, wvl, Dz):
#     N_F = (D1/2)**2 / (wvl*Dz)

#     bigX = x2 / np.sqrt(wvl*Dz)
#     bigY = y2 / np.sqrt(wvl*Dz)
#     alpha1 = -np.sqrt(2) * (np.sqrt(N_F) + bigX)
#     alpha2 = np.sqrt(2) * (np.sqrt(N_F) - bigX)
#     beta1 = -np.sqrt(2) * (np.sqrt(N_F) + bigY)
#     beta2 = np.sqrt(2) * (np.sqrt(N_F) - bigY)

#     sa1, ca1 = scipy.special.fresnel(alpha1)
#     sa2, ca2 = scipy.special.fresnel(alpha2)
#     sb1, cb1 = scipy.special.fresnel(beta1)
#     sb2, cb2 = scipy.special.fresnel(beta2)

#     U = 1/(2*1j) * ((ca2-ca1) + 1j * (sa2 - sa1)) * ((cb2 - cb1) + 1j * (sb2 - sb1))
    

#     return U


def corr2_ft(u1, u2, mask, delta):
    N = (u1.shape)[0]
    c = np.zeros( (N,N) , dtype=np.complex64)
    delta_f = 1/(N*delta)

    U1 = ft2(u1 * mask, delta)
    U2 = ft2(u2 * mask, delta)
    U12corr = ift2(np.conj(U1) * U2, delta_f)

    maskcorr = ift2(abs(ft2(mask, delta))**2, delta_f) * delta**2
    idx = maskcorr != 0
    # How to fix this?
    c[idx] = U12corr[idx] / maskcorr[idx] * mask[idx]
    return c


import os
from datetime import datetime

import os
from datetime import datetime

def create_output_directory(base_dir='outputs'):
    """
    Create a directory structure based on the current date and time.
    Structure: base_dir/YYYY-MM-DD/HH/RUN_<N>

    :param base_dir: Base directory for outputs
    :return: Path to the created output directory
    """
    # Get current date and time
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    hour = now.strftime("%H")
    
    # Create the base path with date and hour
    base_path = os.path.join(base_dir, date, hour)

    # Create the base directories if they don't exist
    if not os.path.exists(base_path):
        os.makedirs(base_path)
        run_number = 1
    else:
        # Find the highest existing run number and increment it
        existing_runs = [d for d in os.listdir(base_path) if d.startswith("RUN_")]
        run_number = max([int(d.split("_")[1]) for d in existing_runs], default=0) + 1

    # Create the run directory
    run_dir = os.path.join(base_path, f"RUN_{run_number}")
    os.makedirs(run_dir, exist_ok=True)
    
    return run_dir



def load_config(file_path='config.json'):
    import json
    with open(file_path, 'r') as file:
        config = json.load(file)
    return config






    
    
if __name__ == "__main__":
    pass