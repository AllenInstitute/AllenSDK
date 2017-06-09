apt-get -yq install \
 makedev \
 fuse \
 libfuse-dev \
 libcurl4-openssl-dev \
 libxml2-dev

git clone https://github.com/s3fs-fuse/s3fs-fuse.git
cd s3fs-fuse
./autogen.sh
./configure
make
make install
