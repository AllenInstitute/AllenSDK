useradd testuser
echo 'testuser:testpassword' | chpasswd
mkdir /home/testuser
cp -R /root/allensdk /home/testuser
chown -R testuser.testuser /home/testuser