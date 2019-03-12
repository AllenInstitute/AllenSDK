#
# A Docker image for running allensdk with neuronal network simulations
#
# docker build --tag alleninstitute/lims2_base:centos7 .
# docker run -it alleninstitute/lims2_base:centos7 /bin/bash
#
FROM alleninstitute/centos7_base
MAINTAINER Tim Fliss <timf@alleninstitute.org>
SHELL [ "/bin/bash", "-c" ]
RUN adduser docker 
RUN echo "export PATH=/usr/local/miniconda/bin:/shared/utils.x86_64/python-2.7/bin:$PATH" >> /home/docker/.bashrc
WORKDIR /home/docker
COPY environment.yml requirements.txt test_requirements.txt internal_requirements.txt Gemfile Gemfile.lock ./

#
# Install Python and NEURON
#
ENV NRN_VER=7.4
ENV NRN=nrn-$NRN_VER NRN_RELEASE=7.4.rel-1370
RUN wget https://repo.continuum.io/miniconda/Miniconda-3.10.1-Linux-x86_64.sh; \
    /bin/bash Miniconda-3.10.1-Linux-x86_64.sh -b -p /usr/local/miniconda; \
    /usr/local/miniconda/bin/conda env create -n lims -f environment.yml; \
    /bin/bash -c "export PATH=/usr/local/miniconda/bin:$PATH; \
    source activate lims; \
    pip install --upgrade pip; \
    pip install -r requirements.txt; \
    pip install -r test_requirements.txt; \
    pip install -r internal_requirements.txt; \
    wget http://www.neuron.yale.edu/ftp/neuron/versions/v7.4/v7.4.rel-1370/nrn-7.4.rel-1370.x86_64.rpm; \
    yum install -y nrn-7.4.rel-1370.x86_64.rpm; \
    rm nrn-7.4.rel-1370.x86_64.rpm"

# RUBY

RUN gpg2 --keyserver hkp://keys.gnupg.net --recv-keys D39DC0E3
RUN curl -L get.rvm.io | /bin/bash -s stable
RUN /bin/bash -c "source /etc/profile.d/rvm.sh; \
    rvm install 1.9.3; \
    rvm use 1.9.3 --default; \
    rvm rubygems current"; \
    /bin/bash -c "source /etc/profile.d/rvm.sh; \
    gem install bundler --no-ri --no-rdoc"; \
    chown docker Gemfile Gemfile.lock; \
    chmod go+rw Gemfile Gemfile.lock; \
    /bin/bash -c "source /etc/profile.d/rvm.sh; \
    rvm use 1.9.3; \
    bundle install"

USER docker
RUN touch .profile
ENV HOME /home/docker
ENV GEM_HOME /home/docker/.gems
#RUN chmod go+w Gemfile.lock; \
#    mkdir -p vendor/gems
USER root
