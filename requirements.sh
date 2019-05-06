sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt-get update
sudo apt-get install python3.6 python3.6-venv python3.6-dev -y
python3.6 -m pip install
pip install virtualenv
python3 -m venv ./venv
source ./venv/bin/activate
pip install -r requirements_python.txt
