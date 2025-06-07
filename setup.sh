#!/bin/bash

# Do all the installation including for Franka-Teach in this repo

# Deoxys
git clone git@github.com:NYU-robot-learning/deoxys_control.git src/deoxys_control
cd src/deoxys_control/deoxys
./InstallPackage  # enter 0.13.3 for the frankalib version when prompted
make -j build_deoxys=1
python -m pip install -e .
python -m pip install -U -r requirements.txt
cd ../../..

# ReSkin
python -m pip install reskin_sensor

# Franka-Env
cd franka-env
python -m pip install -e .
cd ..

git submodule update --init --recursive

# Franka-Teach
cd Franka-Teach
pip install -e .
pip install -r requirements.txt
cd ..

cd co-tracker
git checkout main
pip install -e .
pip install matplotlib flow_vis tqdm tensorboard imageio[ffmpeg]
mkdir -p checkpoints
cd checkpoints
wget https://huggingface.co/facebook/cotracker3/resolve/main/scaled_online.pth
cd ../../

cd dift
pip install xformers
git checkout main
cd ../
