# create conda virtual enviroment and activate it
echo "[*] Creating Conda VENV"
conda create -n pos_tagger
conda activate pos_tagger

# install python libraries
echo "[*] Installing Python libraries"
python -m pip install tensorflow
python -m pip install keras
python -m pip install scikit-learn
python -m pip install matplotlib

echo "[*] All done"