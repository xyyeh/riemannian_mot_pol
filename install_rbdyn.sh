echo "Installing pytorch"
# conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch

echo "Installing Tinyxml2"
pushd ./
git clone https://github.com/leethomason/tinyxml2.git
cd tinyxml2
git checkout 8.0.0
mkdir -p build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=~/anaconda3/envs/rmp/
make -j
make install
popd
rm -rf tinyxml2

echo "Installing eigen and rbdyn"
# build individually with cmake .. -DPYTHON_BINDING_FORCE_PYTHON3=ON
# look for the folder that contains setup.py, run ~/anaconda3/envs/pplanner/bin/pip3 install .

echo "Installing Eigen3ToPython"
pushd ./
git clone https://github.com/jrl-umi3218/Eigen3ToPython.git
cd Eigen3ToPython
git checkout 1.0.2
pip install -r requirements.txt
mkdir -p build
cd build
cmake .. -DPYTHON_BINDING_FORCE_PYTHON3=ON -DCMAKE_INSTALL_PREFIX=~/anaconda3/envs/rmp/
make -j
make install
cd python3
~/anaconda3/envs/rmp/bin/pip3 install .
popd
rm -rf Eigen3ToPython

echo "Installing SpaceVecAlg"
pushd ./
git clone --recursive https://github.com/jrl-umi3218/SpaceVecAlg
cd SpaceVecAlg
git checkout v1.1.0
mkdir -p build
cd build
cmake .. -DPYTHON_BINDING_FORCE_PYTHON3=ON -DBUILD_TESTING=OFF -DCMAKE_INSTALL_PREFIX=~/anaconda3/envs/rmp/
make -j
make install
cd binding/python/sva/python3
~/anaconda3/envs/rmp/bin/pip3 install .
popd
rm -rf SpaceVecAlg

echo "Installing RBDyn"
pushd ./
git clone --recursive https://github.com/jrl-umi3218/RBDyn
cd RBDyn
git checkout v1.3.0
mkdir -p build
cd build
cmake .. -DPYTHON_BINDING_FORCE_PYTHON3=ON -DBUILD_TESTING=OFF -DCMAKE_INSTALL_PREFIX=~/anaconda3/envs/rmp/
make -j
make install
cd binding/python/rbdyn/python3
~/anaconda3/envs/rmp/bin/pip3 install .
popd
rm -rf RBDyn
