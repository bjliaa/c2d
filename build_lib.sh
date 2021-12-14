#!bin/bash

if [ ! -d "./build" ] 
then
    echo "Creating build directory"
    mkdir build  
fi
cd build 

cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=1 ..
sudo make install -j 4 
sudo ldconfig

cd ..