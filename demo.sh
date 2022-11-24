## Training sample
python main.py train --input ./treebanks/UD_Spanish-GSD --output ./output/esp --wordl 16 --sentl 128 --hdim 64 --chdim 32 --activation softmax --bsize 32 --loss categorical_crossentropy --optimizer adam --lr 0.001 --epochs 10

## Decode sample
python main.py decode --input ./demo/sample.txt --output ./demo/predict.txt --modeldir ./output/eng