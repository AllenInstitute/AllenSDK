apt-get -yq install \
 makedev \
 fuse \
 libfuse-dev \
 libcurl4-openssl-dev \
 libxml2-dev

source activate py27

git clone https://github.com/s3fs-fuse/s3fs-fuse.git
cd s3fs-fuse
./autogen.sh
./configure
make
make install

pip install boto3 bokeh awscli
