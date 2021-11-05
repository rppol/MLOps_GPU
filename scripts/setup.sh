#!/bin/bash
export PATH="/home/jenkins/miniconda/bin:$PATH"
source ~/.bashrc 
conda create -n mlops -c rapidsai -c nvidia -c conda-forge rapids=21.10 python=3.8 cudatoolkit=11.2