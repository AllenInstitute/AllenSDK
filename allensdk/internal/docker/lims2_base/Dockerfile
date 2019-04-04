#
# A Docker image for running allensdk with neuronal network simulations
#
# docker build --tag alleninstitute/lims2_base .
# docker run -it alleninstitute/lims2_base /bin/bash
# source ~/env/neurosci/bin/activate
#
FROM ubuntu:trusty
MAINTAINER Tim Fliss <timf@alleninstitute.org>

RUN apt-get update; \
    apt-get install -yq \
    pkg-config \
    libhdf5-dev \
    hdf5-tools \
    libfreetype6-dev \
    unzip \
    wget
RUN wget https://repo.continuum.io/miniconda/Miniconda-3.10.1-Linux-x86_64.sh
RUN /bin/bash Miniconda-3.10.1-Linux-x86_64.sh -b -p /usr/local/miniconda

RUN apt-get install -y -q \
    curl libssl-dev libcurl4-openssl-dev libyaml-dev libffi-dev
RUN apt-get build-dep -y -q python-matplotlib

COPY environment.yml .
RUN /usr/local/miniconda/bin/conda env create -n lims -f environment.yml
ENV PATH /usr/local/miniconda/envs/lims/bin:$PATH

RUN mkdir -p /shared/utils.x86_64; \
    ln -s /usr/local/miniconda/envs/lims /shared/utils.x86_64/python-2.7

ENV NRN_VER=7.4
ENV NRN=nrn-$NRN_VER NRN_RELEASE=7.4.rel-1370

RUN adduser docker --disabled-password --gecos ""
RUN echo "export PATH=/shared/utils.x86_64/python-2.7/bin:$PATH" >> /home/docker/.bashrc

ENV HOME=/home/docker
RUN mkdir -p $HOME/packages
WORKDIR $HOME/packages

RUN wget https://www.neuron.yale.edu/ftp/neuron/versions/v$NRN_VER/v$NRN_RELEASE/nrn-$NRN_RELEASE.tar.gz; \
    tar xzf nrn-$NRN_RELEASE.tar.gz; \
    rm nrn-$NRN_RELEASE.tar.gz

ENV VENV=/usr/local/miniconda/envs/lims
RUN mkdir -p $VENV/build
WORKDIR $VENV/build

RUN apt-get install -yq libncursesw5-dev ncurses-base ncurses-bin ncurses-term

RUN mkdir $NRN; \
    cd $NRN; \
    $HOME/packages/$NRN/configure --with-paranrn --with-nrnpython=$VENV/bin/python --disable-rx3d --without-iv --prefix=$VENV; \
    make; make install; \
    cd src/nrnpython; $VENV/bin/python setup.py install; \
    cd $VENV/bin; ln -s ../x86_64/bin/nrnivmodl


# install ruby
RUN wget http://ftp.ruby-lang.org/pub/ruby/1.9/ruby-1.9.3-p551.tar.gz
RUN tar xzf ruby-1.9.3-p551.tar.gz; \
    rm ruby-1.9.3-p551.tar.gz
RUN cd ruby-1.9.3-p551 && ./configure && make && make install

# install rubygems
RUN wget http://production.cf.rubygems.org/rubygems/rubygems-1.8.24.tgz
RUN tar xvzf rubygems-1.8.24.tgz
RUN cd rubygems-1.8.24 && ruby setup.rb
RUN gem install bundler --no-ri --no-rdoc

RUN apt-get install -y -q \
    curl \
    libcurl4-openssl-dev \
    libpq-dev \
    libmagickwand-dev \
    freetds-dev \
    graphviz \
    graphviz-dev
#    postgresql-libs \
#    postgresql-devel \
WORKDIR /home/docker
COPY Gemfile Gemfile.lock ./
RUN  chown docker Gemfile Gemfile.lock
RUN  chmod go+rw Gemfile Gemfile.lock
USER docker
RUN touch .profile
ENV HOME /home/docker
ENV GEM_HOME /home/docker/.gems
RUN chmod go+w Gemfile.lock
#RUN mkdir -p vendor/gems
RUN  bundle install
