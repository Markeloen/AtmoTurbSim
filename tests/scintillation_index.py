import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.integrate import quad
import matplotlib.animation as animation
import os
from matplotlib.animation import PillowWriter

from core.utils import load_config
from simulator import Simulator

def simulator_scintillation_index_over_time():
    # Create a Simulator object
    config = load_config('config.json')
    simulator = Simulator(config)

    # Intensity Vector for the middle pixel
    intensity_vector = []
    nx_size = config["atmosphere_params"]["phase_screen_size"]
    for i in tqdm(range(simulator.all_steps//10), "Capturing Intensity"):
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

    plt.plot(intensity_vector)
    plt.show()

    print(theo_scintillation_indx)
    print(observed_scintillation_index)

def simulator_scintillation_index_over_time_animate():
    # Create a Simulator object
    config = load_config('config.json')
    simulator = Simulator(config)

    # Intensity Vector for the middle pixel
    nx_size = config["atmosphere_params"]["phase_screen_size"]
    total_steps = simulator.all_steps
    intensity_vector = np.zeros(total_steps)
    Uout_frames = []

    for i in tqdm(range(total_steps), "Capturing Intensity"):
        Uout = simulator.simulate_one_step()
        intensity_vector[i] = np.abs(Uout[nx_size//2, nx_size//2])
        Uout_frames.append(np.abs(Uout))

    output_dir = simulator.output_dir

    # Calculate the theoretical scintillation index
    delta_z = 20e3
    integrand = lambda z: simulator.geometry.cn2(z) * (1 - z/delta_z)**(5/6)
    result, _ = quad(integrand, 0, delta_z)
    theo_scintillation_indx = 0.563 * (2 * np.pi / 1550e-9) ** (7/6) * delta_z ** (5/6) * result

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7), facecolor='#f0f0f0')
    fig.suptitle('Scintillation Index Simulation', fontsize=16)

    line, = ax1.plot([], [], 'r-', linewidth=2)
    ax1.set_xlim(0, total_steps)
    ax1.set_ylim(min(intensity_vector), max(intensity_vector))
    ax1.set_title('Intensity Over Time', fontsize=14)
    ax1.set_xlabel('Time Step', fontsize=12)
    ax1.set_ylabel('Intensity', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_facecolor('#eafff5')

    im = ax2.imshow(Uout_frames[0], cmap='inferno', interpolation='nearest')
    ax2.set_title('Received Wavefront Intensity', fontsize=14)
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

    highlight_pixel = ax2.scatter(nx_size//2, nx_size//2, color='cyan', s=100, edgecolor='white', linewidth=1.5, label='Intensity Measurement Point')
    ax2.legend(loc='upper right', fontsize=10)

    shining_pointer, = ax1.plot([], [], 'bo', markersize=12, markeredgecolor='yellow', markerfacecolor='yellow')

    theoretical_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    observed_text = ax1.text(0.02, 0.88, '', transform=ax1.transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    theoretical_text.set_text(f'Theoretical SI: {theo_scintillation_indx:.3f}')

    def init():
        line.set_data([], [])
        im.set_data(Uout_frames[0])
        shining_pointer.set_data([], [])
        highlight_pixel.set_offsets((nx_size//2, nx_size//2))
        observed_text.set_text('Observed SI: 0.000')
        return line, im, shining_pointer, highlight_pixel, theoretical_text, observed_text

    def update(frame):
        line.set_data(range(frame + 1), intensity_vector[:frame + 1])
        im.set_data(Uout_frames[frame])
        shining_pointer.set_data([frame], [intensity_vector[frame]])
        highlight_pixel.set_offsets((nx_size//2, nx_size//2))
        
        # Calculate observed scintillation index up to the current frame
        mean_intensity = np.mean(intensity_vector[:frame + 1])
        mean_square_intensity = np.mean(intensity_vector[:frame + 1] ** 2)
        observed_scintillation_index = (mean_square_intensity - mean_intensity**2) / mean_intensity**2
        observed_text.set_text(f'Observed SI: {observed_scintillation_index:.3f}')
        
        return line, im, shining_pointer, highlight_pixel, theoretical_text, observed_text

    ani = animation.FuncAnimation(fig, update, frames=range(total_steps), init_func=init, blit=False)
    
    # Save the animation as GIF
    print("Saving animation...")
    writer = PillowWriter(fps=20)
    ani.save(os.path.join(output_dir, 'turbulence_animation.gif'), writer=writer)
    
    plt.close(fig)
    
    print("Animation saved as 'turbulence_animation.gif'")
    print("Saving config file...")
    simulator.save_config()
    print("Config file saved as 'config.json'")

    print(f"Theoretical Scintillation Index: {theo_scintillation_indx:.3f}")


