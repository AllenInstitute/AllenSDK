cd /root
source activate root
apt-get install -y npm
npm install n -g
n 4.5.0
npm install -g configurable-http-proxy
pip install jupyterhub jupyterlab oauthenticator
jupyterhub --generate-config
jupyter serverextension enable --py jupyterlab --sys-prefix
cat "c.JupyterHub.spawner_class = 'dockerspawner.SystemUserSpawner'" >> jupyterhub_config.py
cat 'c.DockerSpawner.container_ip = "0.0.0.0"' >> jupyterhub_config.py
git clone https://github.com/jupyterhub/dockerspawner.git
cd dockerspawner
pip install -r requirements.txt
pip install .
