#!/usr/bin/env bash

if ! which conda > /dev/null; then
      wget https://repo.anaconda.com/archive/Anaconda3-5.2.0-Linux-x86_64.sh -O ~/anaconda.sh
      bash ~/anaconda.sh -b -p $HOME/anaconda3
      echo 'export PATH="$HOME/anaconda3/bin:$PATH"' >> $HOME/.bashrc # add anaconda bin to the environment
      export PATH="$HOME/anaconda3/bin:$PATH"
fi

source $HOME/.bashrc

conda env create -f ./src/environment.yml

source activate pyEPI

python ./src/setup_analysis.py