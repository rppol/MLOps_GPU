#!/bin/bash
wget https://repo.anaconda.com/miniconda/Miniconda3-py38_4.10.3-Linux-x86_64.sh
chmod +x Miniconda3-py38_4.10.3-Linux-x86_64.sh
./Miniconda3-py38_4.10.3-Linux-x86_64.sh -b -u
conda create -n mlops -c rapidsai -c nvidia -c conda-forge rapids=21.10 python=3.8 cudatoolkit=11.2