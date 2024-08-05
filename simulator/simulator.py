import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from tqdm import tqdm
import os
import shutil

from core.propagation import Propagator
from core.geometry import Geometry
from core.propagation.basic_funcs import rect
from core.utils import *



class Simulator:
    def __init__(self, config):
        atmosphere_params = config["atmosphere_params"]
        simulation_params = config["simulation_params"]
        
        self.nx_size = atmosphere_params["phase_screen_size"]
        self.delta = atmosphere_params["delta"]
        self.total_r0 = atmosphere_params["total_r0"]
        self.num_layers = atmosphere_params["num_layers"]
        self.whole_simulation_time = simulation_params["whole_simulation_time"]
        self.per_tick_simulation = simulation_params["per_tick_simulation"]
        self.all_steps = int(self.whole_simulation_time / self.per_tick_simulation)
        
        
        self.geometry = Geometry(
            nx_size=self.nx_size,
            pixel_scale=self.delta,
            r0=self.total_r0,
            number_of_layers=self.num_layers,
            whole_simulation_time=simulation_params["whole_simulation_time"],
            per_tick_simulation=simulation_params["per_tick_simulation"],
            satellite_orbit=simulation_params["satellite_orbit"],
            ground_wind_speed=atmosphere_params["ground_wind_speed"],
            L0=atmosphere_params["L0"]
        )

        
        # Create a timestamped output directory
        self.output_dir = create_output_directory()
        
        # print geometry info
        self.geometry.show_object_info()
        
        self.real_sim_flag = config["real_world_simulation_flag"]
        
        
        # The config file path is always at the root
        self.config_file_path = 'config.json'
        # Creating a propagator class for 20km and 1550nm wvl
        self.propagator = Propagator(self.nx_size, 1550e-9, self.delta, self.delta, self.geometry.layer_height_array)
        
    def save_config(self):
        shutil.copy(self.config_file_path, os.path.join(self.output_dir, 'config.json'))
        
        
        
    def simulate_turb(self):
        steps = self.whole_simulation_time / self.per_tick_simulation
        
        x = np.arange(-self.nx_size/2., self.nx_size/2.) * self.delta
        [x1, y1] = np.meshgrid(x, x) 
        Uin = rect(x1, self.nx_size * self.delta) * rect(y1, self.nx_size * self.delta)

        ps_arr = [layer.scrn for layer in self.geometry.layer_object_array]
        
        for i in range(steps):
            Uout, _, _ = self.propagator.propagate(Uin, ps_arr)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            im1 = ax1.imshow(np.angle(Uout))
            ax1.set_title(f'Phase - Step {i}')
            plt.colorbar(im1, ax=ax1)
            
            im2 = ax2.imshow(np.abs(Uout))
            ax2.set_title(f'Amplitude - Step {i}')
            plt.colorbar(im2, ax=ax2)
            
            plt.tight_layout()
            plt.draw()
            plt.pause(0.01)
            
            ps_arr = self.geometry.move_one_tick_all_layers()
            
            plt.close(fig)  # Close the figure to free up memory
            
            
        plt.close()
        

    def animate_turb(self, steps=100):

        
        if self.real_sim_flag:
            steps = self.all_steps
        
        x = np.arange(-self.nx_size/2., self.nx_size/2.) * self.delta
        [x1, y1] = np.meshgrid(x, x) 
        Uin = rect(x1, self.nx_size * self.delta) * rect(y1, self.nx_size * self.delta)

        ps_arr = [layer.scrn for layer in self.geometry.layer_object_array]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        im1 = ax1.imshow(np.zeros((self.nx_size, self.nx_size)), animated=True)
        im2 = ax2.imshow(np.zeros((self.nx_size, self.nx_size)), animated=True)
        ax1.set_title('Phase')
        ax2.set_title('Amplitude')
        cbar1 = plt.colorbar(im1, ax=ax1)
        cbar2 = plt.colorbar(im2, ax=ax2)

        # Generate frames in advance with a progress bar
        frames = []
        for i in tqdm(range(steps), desc="Generating frames"):
            Uout, _, _ = self.propagator.propagate(Uin, ps_arr)
            frames.append((np.angle(Uout), np.abs(Uout)))
            ps_arr = self.geometry.move_one_tick_all_layers()

        # Animation update function
        def update(frame):
            phase, amplitude = frames[frame]
            
            im1.set_array(phase)
            im2.set_array(amplitude)
            
            # Update color scaling
            im1.set_clim(phase.min(), phase.max())
            im2.set_clim(amplitude.min(), amplitude.max())
            
            # Update colorbar
            cbar1.update_normal(im1)
            cbar2.update_normal(im2)
            
            ax1.set_title(f'Phase - Step {frame}')
            ax2.set_title(f'Amplitude - Step {frame}')
            return im1, im2

        anim = FuncAnimation(fig, update, frames=steps, interval=50, blit=False)
        
        # Save the animation as GIF
        print("Saving animation...")
        writer = PillowWriter(fps=20)
        anim.save(os.path.join(self.output_dir, 'turbulence_animation.gif'), writer=writer)
        
        
        plt.close(fig)

        print("Animation saved as 'turbulence_animation.gif'")
        print("Saving config file...")
        self.save_config()
        print("Config file saved as 'config.json'")


    def simulate_one_step(self):
        

        x = np.arange(-self.nx_size/2., self.nx_size/2.) * self.delta
        [x1, y1] = np.meshgrid(x, x) 
        Uin = rect(x1, self.nx_size * self.delta) * rect(y1, self.nx_size * self.delta)
        
        ps_arr = [layer.scrn for layer in self.geometry.layer_object_array]

        Uout, _, _ = self.propagator.propagate(Uin, ps_arr)
        
        ps_arr = self.geometry.move_one_tick_all_layers()
        return Uout.numpy()
