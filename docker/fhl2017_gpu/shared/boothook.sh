#!/bin/bash
`aws ecr get-login --region us-west-2`
docker run -d -p 80:8000 --privileged=true -v /usr/bin/docker:/usr/bin/docker -v /var/run/docker.sock:/var/run/docker.sock -v /efs/home:/home -v /efs/passwd:/etc/passwd -v /efs/shadow:/etc/shadow -v /s3fs:/data 681657084407.dkr.ecr.us-west-2.amazonaws.com/fhl2017 /bin/bash -c "source /home/shared/nest/bin/nest_vars.sh; /opt/conda/bin/jupyterhub --Spawner.notebook_dir=/home"
