conda update -y conda
source activate root
conda create -n py27 python=2.7 anaconda
source activate root
/opt/conda/envs/py27/bin/python -m ipykernel install
