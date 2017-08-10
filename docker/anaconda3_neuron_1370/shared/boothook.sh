#!/bin/bash
docker run -d -p 80:8000 -v /usr/bin/docker:/usr/bin/docker \
 -v /var/run/docker.sock:/var/run/docker.sock \
 -v /efs/home:/home -v /efs:/efs \
 -v /s3fs:/data 681657084407.dkr.ecr.us-west-2.amazonaws.com/anaconda3_neuron_1370 \
 /bin/bash -c "ln -sf /efs/passwd /etc/passwd; ln -sf /efs/shadow /etc/shadow; source /home/shared/nest/bin/nest_vars.sh; /opt/conda/bin/jupyterhub --Spawner.notebook_dir=nb"