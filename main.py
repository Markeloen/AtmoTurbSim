import sys
import os
import matplotlib.pylab as plt

# Add the project root to the sys.path
script_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(script_directory)
sys.path.append(parent_directory)

# Now you can safely import your modules
from core.layer import Layer
from core.geometry import Geometry
from core.phase_screens import PhaseScreenVonKarman
from core.propagation import Propagator
from simulator import *
from validation_scripts import *
from core.utils import load_config

from tests import *

def run_simulation():
    # Example simulation function
    print("Running simulation...")
    config = load_config('config.json')
    simulator = Simulator(config)
    simulator.animate_turb()

def run_test():
    simulator_scintillation_index_over_time()
    # simulator_scintillation_index_over_time_animate()
    


def main():
    # Main function to orchestrate different parts of your project
    print("Running main script...")
    
    # Example: Running simulation
    run_simulation()

    # Running Test
    run_test()

if __name__ == "__main__":
    main()
