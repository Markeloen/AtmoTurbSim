import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.integrate import quad


from core.utils import load_config
from simulator import Simulator

def simulator_scintillation_index_over_time():
    # Create a Simulator object
    config = load_config('config.json')
    simulator = Simulator(config)

    # Intensity Vector for the middle pixel
    intensity_vector = []
    nx_size = config["atmosphere_params"]["phase_screen_size"]
    for i in tqdm(range(simulator.all_steps), "Capturing Intensity"):
        Uout = simulator.simulate_one_step()
        intensity_vector.append(np.abs(Uout[nx_size//2, nx_size//2]))
    # print(intensity_vector)

    # Calculate the mean intensity
    mean_intensity = np.mean(intensity_vector)

    # Calculate the mean square intensity
    mean_square_intensity = np.mean([i ** 2 for i in intensity_vector])

    # Calculate the observed scintillation index
    observed_scintillation_index = (mean_square_intensity - mean_intensity**2) / mean_intensity**2

    delta_z = 20e3
    integrand = lambda z : simulator.geometry.cn2(z) * (1 - z/delta_z)**(5/6)
    result, _ = quad(integrand, 0 , delta_z)
    theo_scintillation_indx = .563 * (2*np.pi / 1550e-9) ** (7/6) * delta_z ** (5/6) * result


    print(theo_scintillation_indx)
    print(observed_scintillation_index)