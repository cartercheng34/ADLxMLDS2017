if [ $# != 2 ]; then
	echo "Usage: hw2_special.sh [data_directory/] [output_file]"
	exit 1
fi


python3 special.py $1 $2

exit 0