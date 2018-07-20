conda update -y conda
source activate base
conda create -n py36 python=3.6 anaconda
source activate base
/opt/conda/envs/py36/bin/python -m ipykernel install
