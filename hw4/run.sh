if [ $# != 1 ]; then
	echo "Usage: run.sh [testing_file/]"
	exit 1
fi

git clone https://gitlab.com/cartercheng34/MLDS.git

tar zxvf MLDS/skip_thoughts.tgz
rm MLDS/skip_thoughts.tgz
tar zxvf MLDS/model.tgz
rm MLDS/model.tgz

python3 generate.py $1

exit 0