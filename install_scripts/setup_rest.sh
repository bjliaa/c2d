cd ~

sudo apt install -y libopencv-dev 
sudo apt install -y libsdl1.2-dev libsdl-gfx1.2-dev libsdl-image1.2-dev
sudo apt install -y python3-dev python3-opencv python3-pip
sudo apt install -y ffmpeg
sudo apt install -y g++-10
sudo apt install -y clang
sudo apt install -y liblz4-dev

sudo snap install cmake --classic
sudo update-alternatives --set c++ /usr/bin/clang++

pip3 install tensorflow-gpu
pip3 install pandas
