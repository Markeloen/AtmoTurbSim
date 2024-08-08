# Installation Instructions


Install [Miniconda](https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe).

Install [Git](https://git-scm.com/download/win)



# Instructions
1. Verify Python installation
```
 python --version
```
2. Verify Conda installation
```
 conda --version
```
3. Clone the repository
```
 cd /path/to/your/directory
 git clone https://github.com/Markeloen/AtmoTurbSim
 cd AtmoTurbSim
```
4. Set up and activate virtual environment using Conda
```
 conda create --name simulator python=3.9
 conda activate simulator
```
5. Install TensorFlow for GPU run
```
 pip install --upgrade pip
 conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
 pip install "tensorflow<2.11" 

```
6. Verify TensorFlow
```
 python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```
7. Install required packages
```
 pip install -r requirements.txt
```
8. Run the simulator
```
python main.py
```
Outputs are stored in the `/outputs` directory.
