#!/bin/bash

CONDA_ENVS="$(conda env list)"
current_directory=$(pwd)
echo "${current_directory}"

echo "${CONDA_ENVS}"

conda env create -f environment.yml

conda activate darts_albedo


CUR_PYTHON="$(which python)"

# TODO this has to go last for darst cli to work!!!
pip install '.[dev]'


cd ${current_directory}/darts-acquisition
new_dir=$(pwd)
echo "${new_dir}"
pip install '.[dev]'

cd ${current_directory}/darts-ensemble
pip install '.[dev]'

cd ${current_directory}/darts-export
pip install '.[dev]'

cd ${current_directory}/darts-postprocessing
pip install '.[dev]'

cd ${current_directory}/darts-preprocessing
pip install '.[dev]'

cd ${current_directory}/darts-segmentation
pip install '.[dev]'


cd ${current_directory}/darts-superresolution
pip install '.[dev]'


cd ${current_directory}/darts-utils
pip install '.[dev]'
echo Created Environment