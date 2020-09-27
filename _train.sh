#!/bin/bash

# Directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Virtualenv
cd $DIR
virtualenv venv
source venv/bin/activate

# Install dependencies
pip3 install -r requirements.txt

cd $DIR

# Begin experiment
python3.6 main.py
