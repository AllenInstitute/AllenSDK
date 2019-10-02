#
# A Docker image for running allensdk with neuronal network simulations
#
# docker build --tag alleninstitute/lims2_base:centos6 .
# docker run -it alleninstitute/lims2_base:centos7 /bin/bash
# source ~/env/neurosci/bin/activate
#
FROM alleninstitute/centos6_base
MAINTAINER Tim Fliss <timf@alleninstitute.org>

# RUBY

RUN gpg2 --keyserver hkp://keys.gnupg.net --recv-keys D39DC0E3
RUN curl -L get.rvm.io | /bin/bash -s stable
RUN /bin/bash -c "source /etc/profile.d/rvm.sh; \
  rvm install 1.9.3; \
  rvm use 1.9.3 --default; \
  rvm rubygems current"
RUN /bin/bash -c "source /etc/profile.d/rvm.sh; \
  gem install bundler --no-ri --no-rdoc"

# PYTHON

RUN wget https://repo.continuum.io/miniconda/Miniconda-3.10.1-Linux-x86_64.sh; \
    /bin/bash Miniconda-3.10.1-Linux-x86_64.sh -b -p /usr/local/miniconda

COPY environment.yml requirements.txt test_requirements.txt internal_requirements.txt ./
ENV PATH $PATH:/usr/local/miniconda/envs/lims/bin
RUN /usr/local/miniconda/bin/conda env create -n lims -f environment.yml; \
    conda install -n lims -c https://conda.anaconda.org/simpleitk SimpleITK; \
    pip install -r requirements.txt; \
    pip install -r test_requirements.txt; \
    pip install -r internal_requirements.txt

ENV NRN_VER=7.4
ENV NRN=nrn-$NRN_VER NRN_RELEASE=7.4.rel-1370

RUN adduser docker 
RUN echo "export PATH=/shared/utils.x86_64/python-2.7/bin:$PATH" >> /home/docker/.bashrc

ENV HOME=/home/docker
ENV VENV=/usr/local/miniconda/envs/lims
RUN mkdir -p $VENV/build; mkdir -p /home/docker/packages
ENV NRN=nrn-$NRN_VER NRN_RELEASE=7.4.rel-1370

WORKDIR /home/docker/packages

RUN wget https://www.neuron.yale.edu/ftp/neuron/versions/v$NRN_VER/v$NRN_RELEASE/nrn-$NRN_RELEASE.tar.gz;  \
    tar xzf nrn-$NRN_RELEASE.tar.gz; \
    rm nrn-$NRN_RELEASE.tar.gz

WORKDIR $VENV/build

RUN mkdir $NRN; \
    cd $NRN; \
    $HOME/packages/$NRN/configure --with-paranrn --with-nrnpython=$VENV/bin/python --disable-rx3d --without-iv --prefix=$VENV; \
    make; make install; \
    cd src/nrnpython; $VENV/bin/python setup.py install; \
    cd $VENV/bin; rm $VENV/bin/nrnivmodl; ln -s ../x86_64/bin/nrnivmodl

WORKDIR /home/docker
COPY Gemfile Gemfile.lock ./
RUN chown docker Gemfile Gemfile.lock; \
    chmod go+rw Gemfile Gemfile.lock; \
    /bin/bash -c "source /etc/profile.d/rvm.sh; \
    rvm use 1.9.3; \
    bundle install"

USER docker
RUN touch .profile
ENV HOME /home/docker
ENV GEM_HOME /home/docker/.gems
RUN chmod go+w Gemfile.lock; \
    mkdir -p vendor/gems
USER root
