export NRN_VER=7.4
export NRN=nrn-$NRN_VER NRN_RELEASE=7.4.rel-1370
export VENV=/opt/conda/envs/py27

source activate py27
cd $HOME
mkdir packages; cd packages
wget https://www.neuron.yale.edu/ftp/neuron/versions/v$NRN_VER/v$NRN_RELEASE/nrn-$NRN_RELEASE.tar.gz
tar xzf nrn-$NRN_RELEASE.tar.gz
rm nrn-$NRN_RELEASE.tar.gz
cd ..; mkdir $NRN; cd $NRN
$HOME/packages/$NRN/configure --with-paranrn \
 --with-nrnpython=$VENV/bin/python \
 --disable-rx3d --without-iv --prefix=$VENV
make; make install 
cd src/nrnpython; $VENV/bin/python setup.py install
ln -s $VENV/x86_64/bin/nrniv $VENV/bin/nrniv
ln -s $VENV/x86_64/bin/nrnivmodl $VENV/bin/nrnivmodl
