#!/bin/bash

#source activate pytorch_p36

conda create -n maac_env python=3.6

conda activate maac_env

python3.6 -m pip install requirement

cd multiagent-particle-envs

python3.6 -m pip install -e .

cd ../baselines

python3.6 -m pip install -e .

cd ..

python3.6 -m pip install gym==0.9.4

python3.6 -m pip install tensorboardX

python3.6 -m pip install torch seaborn

python3.6 -m pip install wandb



