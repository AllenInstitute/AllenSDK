conda update -y conda
source activate root
pip install -r requirements.txt
pip install -r test_requirements.txt
conda create -n root python=3.6 anaconda
conda create -n py27 python=2.7 anaconda
conda install -n py27 -y -c https://conda.anaconda.org/simpleitk SimpleITK
conda install -n py27 -y mpi4py pyqt cmake gsl libgcc
source activate py27
pip install --upgrade pip
pip install --ignore-installed -r requirements.txt
pip install --ignore-installed -r test_requirements.txt
source activate root
/opt/conda/envs/py27/bin/python -m ipykernel install
