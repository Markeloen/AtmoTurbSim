import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

from Scripts.Geometry import Geometry
from Scripts.Layer import Layer
from Scripts.Utils import *

##Example parameters and setup
# r0 = 20e-2
# layer = Layer(128, 2e-2, r0)

# n = 2  # Number of rows
# fig, axes = plt.subplots(n, 2, figsize=(12, 6))

# # Extract data
# layer.extude_return_column()
# data_xaxis = layer.retrun_modulus_function_xaxis()

# # Configure font size for ticks
# plt.rcParams['xtick.labelsize'] = 8
# plt.rcParams['ytick.labelsize'] = 8

# # Plot initial phase screens and modulus functions
# for i in range(n):
#     # Display the phase screen with a more colorful and clear colormap
#     im = axes[i, 0].imshow(layer.scrn, cmap='plasma')
#     fig.colorbar(im, ax=axes[i, 0], orientation='vertical', shrink=0.6)
#     axes[i, 0].set_title('Phase Screen')
    
#     # Optionally, remove tick labels for a cleaner look
#     axes[i, 0].set_xticklabels([])
#     axes[i, 0].set_yticklabels([])

#     # Plot modulus function with clearly distinguished line styles and markers
#     axes[i, 1].plot(data_xaxis[2], data_xaxis[0], 'r-', label='Modulus 1')
#     axes[i, 1].plot(data_xaxis[2], data_xaxis[1], 'b--', label='Modulus 2')
#     axes[i, 1].legend(loc='upper right')
#     axes[i, 1].set_xlabel('X-axis')
#     axes[i, 1].set_ylabel('Modulus Value')
#     axes[i, 1].set_title('Modulus Function')

# # Improve overall aesthetics
# plt.tight_layout()
# plt.show(block=False)

# # Update plots in a loop
# for _ in range(1000):  # Modify this range as necessary
#     layer.extude_return_column()
#     for i in range(n):
#         # Update data
#         data_xaxis = layer.retrun_modulus_function_xaxis()
        
#         # Update phase screen
#         axes[i, 0].images[0].set_data(layer.scrn)
        
#         # Optionally, remove tick labels to keep the updated phase screen clean
#         axes[i, 0].set_xticklabels([])
#         axes[i, 0].set_yticklabels([])

#         # Update modulus function plots
#         axes[i, 1].lines[0].set_ydata(data_xaxis[0])
#         axes[i, 1].lines[0].set_xdata(data_xaxis[2])
#         axes[i, 1].lines[1].set_ydata(data_xaxis[1])
#         axes[i, 1].lines[1].set_xdata(data_xaxis[2])
        
#     plt.pause(0.1)  # Adjust pause time as needed for your update frequency


geometry = Geometry(128, 2e-2)
geometry.show_object_info()
geometry.move_one_tick_all_layers()
# show_geomerty_layers(geometry)

