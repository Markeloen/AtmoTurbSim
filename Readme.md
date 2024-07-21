# Atmosphere Simulator

This project contains a comprehensive set of modules for simulating and validating different systems. The project is organized into several main directories, each serving a specific purpose.



## Getting Started

### Prerequisites

Ensure you have the following installed:
- Python 3.8+
- Required Python packages (listed in `requirements.txt`)

### Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/Markeloen/AtmoTurbSim
    cd yourrepository
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

### Running the Project

#### Main Script

The `main.py` file serves as the primary entry point for running your application. It can orchestrate different parts of your project, such as running simulations, validations, or other operations.

To run the main script:
```sh
python main.py
```
### Directory Overview
core/: Contains the core modules and functionalities of the project.

layer/: Layer-related functionalities.

geometry/: Geometry-related functionalities.

phase_screen/: Phase screen-related functionalities.

propagation/: Propagation-related functionalities.

utils.py: Utility functions.

simulator/: Contains simulation-specific modules and functionalities.

validation_scripts/: Contains scripts for validating different parts of the project.

scripts/: Contains scripts to run simulations, validations, or other operations.

legacy_scripts/: Contains legacy scripts for backward compatibility or reference.

tests/: Contains unit tests for the modules and functionalities.

outputs/: Contains all output files from simulations, validations, etc.

simulations/: Contains output files from simulations.

validations/: Contains output files from validation scripts.

requirements.txt: Lists the required Python packages for the project.

main.py: The main entry point of the application.

README.md: Project documentation.

#Contributing
Feel free to submit issues and enhancement requests. For contributing, follow these steps:

Fork the repository.
Create a new branch (git checkout -b feature/your-feature).
Commit your changes (git commit -am 'Add some feature').
Push to the branch (git push origin feature/your-feature).
Create a new Pull Request.
