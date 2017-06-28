apt-get -yq install \
 makedev \
 fuse \
 libfuse-dev \
 libcurl4-openssl-dev \
 libxml2-dev

source activate py27
pip install boto3 bokeh=0.12.6 awscli
conda install -c nicholasc dipde
