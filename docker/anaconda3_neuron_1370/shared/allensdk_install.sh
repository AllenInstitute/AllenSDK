cd /root
wget https://github.com/AllenInstitute/AllenSDK/archive/v0.13.2.tar.gz
tar xvzf v0.13.2.tar.gz
mv AllenSDK-0.13.2 allensdk
cd allensdk
source activate root
pip install --ignore-installed -r requirements.txt
pip install --ignore-installed -r test_requirements.txt
pip install .
source activate py27
pip install --ignore-installed -r requirements.txt
pip install --ignore-installed -r test_requirements.txt
pip install .
