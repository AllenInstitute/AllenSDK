source activate py27
CUDA_REPO_PKG=cuda-repo-ubuntu1404_7.5-18_amd64.deb && wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/$CUDA_REPO_PKG && dpkg -i $CUDA_REPO_PKG
pip install tensorflow-gpu keras