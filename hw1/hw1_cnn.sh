#!/bin/bash
# 
# 2017Fall MLDS HW1

if [ $# != 2 ]; then
	echo "Usage: hw1_rnn.sh [data_directory/] [output_file]"
	exit 1
fi

wget --no-check-certificate "https://drive.google.com/uc?export=download&id=0Bzf1OKQxrRkTcUpTVm93TnZDS2c" -O model.zip
unzip model.zip

python3 test_cnn.py $1 $2

exit 0