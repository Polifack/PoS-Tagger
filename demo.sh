## Training sample
python main.py train --input ./treebanks/UD_English-EWT --output ./output --wordl 16 --sentl 128 --hdim 64 --chdim 32 --activation softmax --bsize 32 --loss categorical_crossentropy --optimizer adam --lr 0.001 --epochs 10

## Decode sample
python main.py decode --input ./sample.txt --output ./predict.txt --modeldir ./output