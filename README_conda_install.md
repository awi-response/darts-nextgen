# DARTS nextgen conda installation instructions

In some cases uv might not be ideal for managing dependencies, since it does not install cuda related 
dependencies, but expects them to be installed on the host machine. These instructions for using conda
provide an alternative. 

To install using conda, do the following.

First, create a conda environment : 

` conda create -n rts_126 python=3.12.9`

The install these dependencies using pip

`pip install -e ./darts-acquisition `

`pip install -e ./darts-ensemble`

`pip install -e ./darts-export`

`pip install -e ./darts-postprocessing`

`pip install -e ./darts-preprocessing`

`pip install -e ./darts-segmentation`

`pip install -e ./darts-superresolution`

`pip install -e ./darts-utils`

After that, install the following packages using conda

`conda install -c conda-forge cupy=12.0.0 cudatoolkit=12.6
`
`conda install conda-forge::google-cloud-sdk`

After that, do the following

`pip install ".[cuda126]"`

At this point, you should be able to run the pipeline. Checking by typign 

`darts --help`

do make sure darts is installed, and 

`nvcc --version`

to make sure cuda related dependencies are installed.

If you make changes to any of the code, you will need to reinstall the packages you changed. 
