for USER in timf davidf nikah nicholasc lukec forrestc nileg justink
do
  useradd $USER
  echo "$USER:log${USER}in" | chpasswd
  chsh -s /bin/bash $USER
  mkdir /home/$USER
  echo "AKIAJ3HPE46FBJ4XMTDQ:+S2t1Dxs+w9JAMAyPw4PKgG45Epv7SoEWk+G2xOd" > /home/$USER/.passwd-s3fs
  cp -R /root/allensdk /home/$USER
  chown -R ${USER}.$USER /home/$USER
done