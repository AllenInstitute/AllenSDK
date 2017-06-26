#!/bin/bash
USER=$1
useradd $USER
echo "$USER:log${USER}in" | chpasswd
chsh -s /bin/bash $USER
mkdir /home/$USER
cp -R /root/allensdk /home/$USER
chown -R ${USER}.$USER /home/$USER
cat source /home/shared/nets/bin/nest_vars.sh >> ~/.bashrc
