#!/bin/bash
function print_header(){
    printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' - 
	echo $1
	printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
}

# Directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Virtualenv
cd $DIR
virtualenv venv
source venv/bin/activate

# Install dependencies
pip3 install -r requirements.txt

# Begin experiment
print_header "Training network"
cd $DIR
python3.6 main.py \
--prefix "" \
--n_hidden 128 \
--test ${1:-0}
