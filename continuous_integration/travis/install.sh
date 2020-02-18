set -xe

#!/usr/bin/env bash
# Install miniconda
wget https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"
export BOTO_CONFIG=/dev/null
conda config --set always_yes yes --set changeps1 no --set remote_max_retries 10

# Create conda environment
conda env create -q -n test-environment -f $ENV_FILE
source activate test-environment

# Install
python -m pip install --quiet --no-deps -e .
echo conda list
conda list

set +xe
