conda update -y conda
source activate base
conda create -n py27 python=2.7 anaconda
source activate py27
pip install -y pipenv
/opt/conda/envs/py27/bin/python -m ipykernel install
