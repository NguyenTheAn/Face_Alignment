mkdir build
cd build
cmake ..
make -j4
export LD_LIBRARY_PATH=/home/ligirk/Workplace/facial_alignment/opencv/lib/:$LD_LIBRARY_PATH
./FaceStandarizer $1