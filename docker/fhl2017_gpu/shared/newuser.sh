#!/bin/bash
USER=$1
useradd $USER
echo "$USER:log${USER}in" | chpasswd
chsh -s /bin/bash $USER
#mkdir /home/$USER
#cp -R /home/user/* /home/$USER
#echo source /home/shared/nest/bin/nest_vars.sh >> /home/${USER}/.bashrc
chown -R ${USER} /home/$USER
