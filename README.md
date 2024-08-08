# Installation Instructions
Install [Python 3.12.4](https://www.python.org/downloads/release/python-3124/).

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
 conda create --name simulator python=3.12.4
 conda activate simulator
```
5. Install TensorFlow for GPU run
```
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
pip install tensorflow==2.10

```

pip install virtualenv
python -m venv venv
source venv/bin/activate
# Install requirements
pip install -r requirements.txt
# Run the main script
python main.py
```
