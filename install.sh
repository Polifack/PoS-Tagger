# create conda virtual enviroment and activate it
echo "[*] Creating Conda VENV"
conda create -n pos_tagger python=3.8
conda activate pos_tagger

# install python libraries
echo "[*] Installing Python libraries"
pip install tensorflow
pip install keras
pip install conllu

echo "[*] All done"