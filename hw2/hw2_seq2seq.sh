if [ $# != 3 ]; then
	echo "Usage: hw2_seq2seq.sh [data_directory/] [output_file] [peer_output_file]"
	exit 1
fi


python3 test.py $1 $2 $3

exit 0