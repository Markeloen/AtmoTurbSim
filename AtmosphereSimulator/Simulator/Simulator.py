import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from tqdm import tqdm

from propagation.Propagator import Propagator
from Scripts.Geometry import Geometry
from propagation.basic_funcs import rect



class Simulator:
    def __init__(self, nx_size, delta):
        self.nx_size = nx_size
        self.delta = delta
        self.geometry = Geometry(nx_size, delta)
        # Creating a propagator class for 20km and 1550nm wvl
        self.propagator = Propagator(nx_size, 1550e-9, delta, delta, self.geometry.layer_height_array)
        
        
    def simulate_turb(self):
        steps = 1000
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
        
    def animate_turb(self, steps=1000):
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
        anim.save('turbulence_animation.gif', writer=writer)
        
        plt.close(fig)

        print("Animation saved as 'turbulence_animation.gif'")