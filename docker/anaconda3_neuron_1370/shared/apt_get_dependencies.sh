apt-get update
apt-get -yq install \
 pkg-config \
 libfreetype6-dev \
 build-essential \
 make \
 automake \
 libtool \
 bison \
 flex \
 libxext-dev \
 libncurses-dev \
 libhdf5-dev \
 wget \
 openmpi-bin \
 libopenmpi-dev \
 alien \
 unzip \
 vim \
 libffi-dev \
 libopenjp2-7

wget http://www.cmake.org/files/v3.4/cmake-3.4.1.tar.gz 
tar -xvzf cmake-3.4.1.tar.gz 
cd cmake-3.4.1/ 
./configure 
make
make install
update-alternatives --install /usr/bin/cmake cmake /usr/local/bin/cmake 1 --force