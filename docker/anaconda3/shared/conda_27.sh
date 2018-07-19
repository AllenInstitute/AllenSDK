conda update -y conda
source activate base
conda create -n py27 python=2.7 anaconda
source activate base
/opt/conda/envs/py27/bin/python -m ipykernel install
