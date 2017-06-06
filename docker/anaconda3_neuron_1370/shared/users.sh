for USER in timf davidf nikah nicholasc lukec forrestc
do
  useradd $USER
  echo "$USER:log${USER}in" | chpasswd
  mkdir /home/$USER
  cp -R /root/allensdk /home/$USER
  chown -R ${USER}.$USER /home/$USER
done