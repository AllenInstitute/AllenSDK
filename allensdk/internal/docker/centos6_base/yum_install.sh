yum -y groupinstall 'Development Tools'
yum -y install epel-release
yum install -y unzip wget \
 gcc-c++ patch readline readline-devel zlib zlib-devel \
 libyaml-devel libffi-devel openssl-devel make \
 bzip2 autoconf automake libtool bison iconv-devel sqlite-devel \
 kernel-devel ImageMagick-devel bzip2-devel libcurl libcurl-devel \
 openssl-devel \
 libevent-devel \
 libffi-devel \
 glib2-devel \
 libjpeg-devel \
 mysql-devel \
 postgresql-devel \
 ncurses-devel \
 readline \
 readline-devel \
 sqlite-devel \
 openssl \
 openssl-devel \
 libxml2-devel \
 libxslt-devel \
 zlib-devel \
 boost boost-devel \
 tinyxml tinyxml-devel \
 glibc-devel.i686 libstdc++-devel.i686 gcc-g++ \
 hdf5 hdf5-devel libuuid-devel \
 libpng-devel \
 which \
 gcc \
 g++ \
 make \
 cmake \
 automake \
 autoconf \
 findutiles \
 apr-devel \
 apr-util-devel \
 ruby-devel \
 libcurl-openssl-devel
yum clean all
mkdir -p /shared/bioapps/cmake-latest/bin
ln -s /usr/bin/cmake /shared/bioapps/cmake-latest/bin/cmake