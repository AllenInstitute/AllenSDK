# DOCKER-VERSION 1.11.1
#
# docker build --tag alleninstitute/jupyterhub:aws .
# docker run -it alleninstitute/jupyterhub:aws jupyter-notebook
#
FROM alleninstitute/allensdk:anaconda3_neuron_1370

MAINTAINER Tim Fliss <timf@alleninstitute.org>

RUN conda create -n py27 python=2.7 anaconda

RUN /bin/bash --login -c "cd /root; \
    wget https://github.com/AllenInstitute/AllenSDK/archive/v0.13.1.tar.gz; \
    tar xvzf v0.13.1.tar.gz; \
    mv AllenSDK-0.13.1 allensdk; \
    cd allensdk; \
    pip install -r requirements.txt; \
    pip install -r test_requirements.txt; \
    pip install .; \
    ipython kernel install; \
    source activate root; \
    /opt/conda/envs/py27/bin/python -m ipykernel install"
    

RUN apt-get install -y npm vim; \
    npm install n -g; \
    n 4.5.0; \
    npm install -g configurable-http-proxy; \
    pip install jupyterhub oauthenticator; \
    jupyterhub --generate-config; \
    cat "c.JupyterHub.spawner_class = 'dockerspawner.SystemUserSpawner'" >> jupyterhub_config.py; \
    cat 'c.DockerSpawner.container_ip = "0.0.0.0"' >> jupyterhub_config.py; \
    git clone https://github.com/jupyterhub/dockerspawner.git; \
    cd dockerspawner; \
    
    pip install -r requirements.txt; \
    pip install .

RUN useradd testuser; \
    echo 'testuser:testpassword' | chpasswd; \
    mkdir /home/testuser; \
    cp -R /root/allensdk /home/testuser; \
    chown -R testuser.testuser /home/testuser

EXPOSE 8888

CMD ["jupyterhub", "--ip=0.0.0.0", "--Spawner.notebook_dir=allensdk/doc_template/examples/nb"]