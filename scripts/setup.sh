#!/bin/bash
source ~/.bashrc
conda activate mlops
cp -R /home/nvidiatest/mlops_blog /var/lib/jenkins/workspace/Train
ls -a