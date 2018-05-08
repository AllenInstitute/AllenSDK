#!/bin/bash
`aws ecr get-login --region us-west-2 --no-include-email`
docker run -d --rm -p 80:8000 -v /usr/bin/docker:/usr/bin/docker -v /var/run/docker.sock:/var/run/docker.sock -v /efs/home:/home -v /efs/passwd:/etc/passwd -v /efs/shadow:/etc/shadow -v /s3fs:/data $(ls /dev/nvidia* | xargs -I{} echo '--device={}') $(ls /usr/lib/x86_64-linux-gnu/{libcuda,libnvidia}* | xargs -I{} echo '-v {}:{}:ro')  681657084407.dkr.ecr.us-west-2.amazonaws.com/fhl2017:gpu /bin/bash -c "source /home/shared/nest/bin/nest_vars.sh; /opt/conda/bin/jupyterhub --Spawner.notebook_dir=/home"
#docker run -d busybox
