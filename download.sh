#MNIST dataset
DIRECTORY="dataset/mnist"

if [ ! -d "dataset" ]; then
  mkdir "dataset"
fi
if [ ! -d "$DIRECTORY" ]; then
  mkdir "$DIRECTORY"
  echo "Created $DIRECTORY directory."
fi

BASE='http://yann.lecun.com/exdb/mnist/'
URLS='train-images-idx3-ubyte.gz train-labels-idx1-ubyte.gz t10k-images-idx3-ubyte.gz t10k-labels-idx1-ubyte.gz'

for i in $URLS; do
	echo "$i"
	wget -c -O "$DIRECTORY/$i" "$BASE$i"
done

echo "Unziping downloaded files..."
gunzip $DIRECTORY/*.gz
echo "All files downloaded to $DIRECTORY."


# EIGEN
if [ ! -d "lib" ]; then
  mkdir "lib"
  echo "Created lib directory."
fi

wget -c -O "lib/eigen.tar.bz2" "http://bitbucket.org/eigen/eigen/get/3.3.3.tar.bz2"

echo "Unziping Eigen..."
tar xjf "lib/eigen.tar.bz2" -C "lib"
mv lib/eigen-* lib/eigen
rm  "lib/eigen.tar.bz2"

echo "Eigen extracted to directory lib/eigen."
