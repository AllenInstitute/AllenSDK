conda update -y conda
source activate root
pip install -y pipenv
conda create -n py27 python=2.7 anaconda
source activate root
pip install -y pipenv
/opt/conda/envs/py27/bin/python -m ipykernel install
