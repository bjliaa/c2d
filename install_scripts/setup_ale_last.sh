cd ~
git clone https://github.com/mgbellemare/Arcade-Learning-Environment.git
cd Arcade-Learning-Environment
mkdir build && cd build
cmake -DUSE_SDL=OFF -DBUILD_CPP_LIB=ON -DBUILD_PYTHON=OFF ..
sudo make -j 10
sudo make install