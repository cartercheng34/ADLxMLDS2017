#!/bin/bash
# 
# 2017Fall MLDS HW1

if [ $# != 2 ]; then
	echo "Usage: hw1_rnn.sh [data_directory/] [output_file]"
	exit 1
fi

wget --no-check-certificate "https://drive.google.com/uc?export=download&id=0Bzf1OKQxrRkTZlRuNUNQVGplMGc" -O model2.zip
unzip model2.zip
wget --no-check-certificate "https://drive.google.com/uc?export=download&id=0Bzf1OKQxrRkTNGVkSzlUM3NzN3c" -O model3.zip
unzip model3.zip

python3 ensemble.py $1 $2

exit 0